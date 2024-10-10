"""
/*
* Copyright 2018-2020 Philippe Tillet
* Copyright 2020-2022 OpenAI
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

Softmax
===============

Source: https://github.com/ROCm/triton/blob/main_perf/python/perf-kernels/softmax.py
"""

import argparse
import torch
import sys
import pytest

import triton
import triton.language as tl


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')


def get_cuda_autotune_config():
    return [
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
    ]


def get_hip_autotune_config():
    return [
        triton.Config({'waves_per_eu': 1}, num_warps=4, num_stages=1),
        triton.Config({'waves_per_eu': 1}, num_warps=8, num_stages=1),
        triton.Config({'waves_per_eu': 1}, num_warps=16, num_stages=1),
        triton.Config({'waves_per_eu': 2}, num_warps=4, num_stages=1),
        triton.Config({'waves_per_eu': 2}, num_warps=8, num_stages=1),
        triton.Config({'waves_per_eu': 2}, num_warps=16, num_stages=1),
        triton.Config({'waves_per_eu': 4}, num_warps=4, num_stages=1),
        triton.Config({'waves_per_eu': 4}, num_warps=8, num_stages=1),
        triton.Config({'waves_per_eu': 4}, num_warps=16, num_stages=1),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


@triton.autotune(configs=get_autotune_config(), key=['n_rows', 'n_cols'], use_cuda_graph=True)
@triton.jit
def softmax_kernel_online(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols,
                          BLOCK_SIZE: tl.constexpr):

    row_start = tl.program_id(0)
    row_idx = row_start

    #loop 1, find max and sum
    m = -float('inf')  #Initial value of max
    row_sum = 0.0
    row_start_ptr = input_ptr + row_idx * input_row_stride
    for b in tl.range(0, n_cols, BLOCK_SIZE):
        col_offsets = b + tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row_block = tl.load(input_ptrs, mask=mask, other=-float('inf'), cache_modifier=".cg")  #load block
        m_p = tl.max(row_block, axis=0)  #find block max
        m_p = tl.maximum(m, m_p)  #Find new max across all blocks so far
        row_sum = row_sum * tl.exp(m - m_p)  #Adjust previous sum
        row_sum += tl.sum(tl.exp(row_block - m_p))  #Add to exponentiated sum of this block
        m = m_p  #save max

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    #Loop 2
    for b in tl.range(0, n_cols, BLOCK_SIZE):
        col_offsets = b + tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row_block = tl.load(input_ptrs, mask=mask, other=-float('inf'), cache_modifier=".cg")  #load block
        #subtract, exponentiate and divide by sum
        softmax_output = tl.exp(row_block - m) / row_sum
        #store
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    y = torch.empty_like(x)

    num_programs = n_rows

    grid = lambda meta: (num_programs, )
    softmax_kernel_online[grid](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE,
    )

    return y


def run_softmax(M, N):
    print(f"Running Softmax on shape ({M},{N})")
    torch.manual_seed(0)
    x = torch.randn(M, N, device='cuda')
    y_triton = softmax(x)

    return y_triton


#Accuracy
@pytest.mark.parametrize('M, N', [(1823, 781), (1, 1), (128, 1), (1, 128), (8192, 8192), (4096, 8192), (359, 1),
                                  (1, 359), (1, 131072), (1, 89999)])
def test_softmax(M, N):
    torch.manual_seed(0)
    x = torch.randn(M, N, device='cuda')
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)


#Benchmark
arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def test_benchmark():
    dtype = "fp16"
    config = []
    x_vals_list = [i for i in range(1024, 65536, 2048)]
    mn_args = {'M': 1}
    plot_name = str("softmax-performance_" + dtype + "_M" + str(1) + "_N" + str(1024) +
                    "-" + str(65536) + "-" + str(2048))
    x_names = ['N']
    dtype = arg_to_torch_dtype[dtype]

    config.append(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=[
                "Triton",
                "Torch",
            ],
            styles=[('blue', '-'), ('green', '-')],
            ylabel="GB/s",
            plot_name=plot_name,
            args=mn_args,
        ))

    @triton.testing.perf_report(config)
    def benchmark(M, N, provider):
        x = torch.randn(M, N, device='cuda', dtype=dtype)
        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)
        if provider == 'torch':
            ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
        if provider == 'triton':
            ms = triton.testing.do_bench(lambda: softmax(x))
        gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    benchmark.run(save_path="./perf-artifacts/softmax", show_plots=True, print_data=True)
