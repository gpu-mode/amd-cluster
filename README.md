# AMD Kernels

This repo contains customized kernels for AMD Instinct series GPUs.
Please make sure your Triton compiler is v2.1 or later, and is from the OpenAI Triton repository
[here](https://github.com/openai/triton). To install Triton, please see
[these](https://github.com/openai/triton/tree/main?tab=readme-ov-file#install-from-source) instructions.
You can also install in your python venv using latest wheels:
`pip install --pre pytorch-triton-rocm torch --index-url https://download.pytorch.org/whl/nightly/rocm6.2`

## How to add your own benchmark

This repo is configured with a custom Github runner donated by AMD. You can queue jobs to this runner by either merging your code or by opening a pull request. We don't need to merge your code for you to run benchmarks.

The main things you need to run your own benchmark
1. In `kernels/` create a new file that must start with the name `test_`. This is because we use `pytest` to discover your kernel
2. If you want your benchmark results to persist in a Github Artifact, we recommend using the builtin Triton `benchmark.run(save_path="./perf-artifacts/your_kernel", show_plots=True, print_data=True)`
3. In your PR, if you don't want to run testing on all the kernels, you can specify a specific kernel you want
to test by adding a line like the following to your PR description: `ci-exactly: <test-file-name.py>` as seen
in this PR: [Example](https://github.com/gpu-mode/amd-cluster/pull/3)

Have fun! We intend for this to be a social repo, if you have any other requests for things we could do better please let us know!

## Existing benchmarks

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

# Dev roadmap

CI changes
* [ ] Doesn't make sense to run the full benchmark suite on each PR, instead only run changed files
* [ ] Considering we have a node, running the tests sequentially seems like a miss, instead should allocate a test to a free gpu. Investigate tech like `pytest-xdist`
* [ ] Setting up triton env takes a few min, we should cache this since it almost never changes

UX changes
Instead of submitting jobs via Github we could do it via Discord. UX would be a
* [ ] user submits a kernel.py in #rocm channel on discord.gg/gpumode and that gets picked up a Discord bot
* [ ] Given a script, use the bot to automatically open a PR for benchmarking. This can be done thanks to tools like https://github.com/PyGithub/PyGithub 
* [ ] Once the triggered Github action is complete the bot can reply to the original user message with a link to the generated Github artifact. If the job fails then the bot should link to the failed Github Action
* [ ] Nice to have would be to give users a sense of their position on the queue  
