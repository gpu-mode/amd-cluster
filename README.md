# AMD Kernels

This repo contains customized kernels for AMD Instinct series GPUs.
Please make sure your Triton compiler is v2.1 or later, and is from the OpenAI Triton repository
[here](https://github.com/openai/triton). To install Triton, please see
[these](https://github.com/openai/triton/tree/main?tab=readme-ov-file#install-from-source) instructions.
You can also install in your python venv using latest wheels:
`pip install --pre pytorch-triton-rocm torch --index-url https://download.pytorch.org/whl/nightly/rocm6.2`

## `test_flash-attention.py`

This script contains the Flash Attention kernel with the following support

- Arbitrary Q and KV sequence lengths, and arbitrary head sizes
- Autoregressive or "causal" masking
- Flash Attention v2 with variable sequence lengths
- Multi and Grouped Query attention
- ALiBi bias
- Matrix bias

These are currently supported for the forward kernel only.

## `test_matmul.py`

This script contains the GEMM kernel that supports int8, int32, fp16,
fp32, bf16 datatypes.

## `test_softmax.py`

Kernel that implements Softmax over a row of tensor.

## `test_rmsnorm.py`

Kernel that implements RMS Norm over a row of tensor.

## `test_layernorm.py`
Kernel that implements Layer Normalization over a row on tensor

## `test_dotproduct.py`
Kernel that implements the dot product of two vectors