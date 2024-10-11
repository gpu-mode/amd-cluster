"""
rmsnorm
===============

Source: https://github.com/ROCm/triton/blob/b36e072fc52e4a9d2222460519f4a8ff669d0f7e/python/perf-kernels/rmsnorm.py
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
def rms_kernel(output_ptr, input_ptr, g_ptr, input_row_stride, output_row_stride, n_rows, n_cols, epsilon,
               BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        input_ptrs = row_start_ptr + col_offsets
        input_ptrs = tl.multiple_of(input_ptrs, (16, ))
        row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
        g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0, cache_modifier=".cg")
        row_norm = row * row  #square each value
        row_norm = tl.sum(row_norm, axis=-1)  #sum across columns(axis=-1)
        row_norm = row_norm / n_cols  #divide by n_cols
        row_norm = row_norm + epsilon  #add epsilon
        row_norm = tl.rsqrt(row_norm)  #take rsqrt, this is normalization value
        rms_norm = row * row_norm  #multiply each x by normalization value
        rms_norm = rms_norm * g  #element wise multiplication with g

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        output_ptrs = tl.multiple_of(output_ptrs, (16, ))
        tl.store(output_ptrs, rms_norm, mask=mask)


def rmsnorm(x, epsilon=1e-6):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    y = torch.empty_like(x, device='cuda')
    g = torch.ones((1, n_cols), device='cuda')

    num_programs = n_rows
    grid = lambda meta: (num_programs, )
    rms_kernel[grid](y, x, g, x.stride(0), y.stride(0), n_rows, n_cols, epsilon, BLOCK_SIZE)

    return y


def run_rmsnorm(M, N):
    torch.manual_seed(0)
    x = torch.randn(M, N, device='cuda')
    y_triton = rmsnorm(x)

    return y_triton


@pytest.mark.parametrize('M, N', [
    (1, 4),
    (2, 10),
    (8192, 4096),
    (4096, 8192),
    (1, 8192),
    (873, 1245),
])
def test_rmsnorm(M, N):
    torch.manual_seed(0)
    x = torch.randn(M, N, device='cuda')
    y_triton = rmsnorm(x)

    rms_norm = torch.nn.RMSNorm(N, device='cuda')
    y_torch = rms_norm(x)

    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)


#Benchmark
arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def torch_rmsnorm(x):
    M, N = x.shape
    rms_norm = torch.nn.RMSNorm(N, device='cuda')
    y_torch = rms_norm(x)

    return y_torch


def test_benchmark():
    config = []
    dtype = "fp16"
    x_vals_list = [i for i in range(8192, 32768, 1024)]
    mn_args = {'M': 1}
    x_names = ['N']
    plot_name = str("rmsnorm-performance_" + dtype + "_M" + str(1) + "_N" + str(8192) +
                    "-" + str(32768) + "-" + str(1024))

    dtype = arg_to_torch_dtype[dtype]

    print(plot_name)
    config.append(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=["Triton", "Torch"],
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
            ms = triton.testing.do_bench(lambda: torch_rmsnorm(x))
        if provider == 'triton':
            ms = triton.testing.do_bench(lambda: rmsnorm(x))
        gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    benchmark.run(save_path="./perf-artifacts/rmsnorm", show_plots=True, print_data=True)
