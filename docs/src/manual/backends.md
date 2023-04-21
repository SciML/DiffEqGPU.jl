# Compute Backends (GPU Choices)

DiffEqGPU.jl supports a multitude of different GPU devices. These must be chosen during the
construction of the `EnsembleGPUArray` and `EnsembleGPUKernel` construction and correpond
to the compute backends of [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).
The choices for backends are:

  - `CUDA.CUDABackend()`: For NVIDIA GPUs via code generation for CUDA kernels.
  - `AMDGPU.ROCBackend()`: For AMD GPUs via code generation for ROCm kernels.
  - `oneAPI.oneAPIBackend()`: For Intel GPUs via code generation for OneAPI kernels.
  - `Metal.MetalBackend()`: For Apple Silicon (M-Series such as M1 or M2) via code generation
    for Metal kernels.

This is used for example like `EnsembleGPUKernel(oneAPI.oneAPIBackend())` to enable the
computations for Intel GPUs. The choice of backend is mandatory and requires the installation
of the respective package. Thus for example, using the OneAPI backend requires that the
user has successfully installed [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl) and has
an Intel GPU.
