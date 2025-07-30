# DiffEqGPU

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/DiffEqGPU/stable/)
[![Build status](https://badge.buildkite.com/409ab4d885030062681a444328868d2e8ad117cadc0a7e1424.svg)](https://buildkite.com/julialang/diffeqgpu-dot-jl)

[![codecov](https://codecov.io/gh/SciML/DiffEqGPU.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/DiffEqGPU.jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

This library is a component package of the DifferentialEquations.jl ecosystem. It includes
functionality for making use of GPUs in the differential equation solvers.


## The two ways to accelerate ODE solvers with GPUs

There are two very different ways that one can
accelerate an ODE solution with GPUs. There is one case where `u` is very big and `f`
is very expensive but very structured, and you use GPUs to accelerate the computation
of said `f`. The other use case is where `u` is very small but you want to solve the ODE
`f` over many different initial conditions (`u0`) or parameters `p`. In that case, you can
use GPUs to parallelize over different parameters and initial conditions. In other words:

| Type of Problem                           | SciML Solution                                                                                           |
|:----------------------------------------- |:-------------------------------------------------------------------------------------------------------- |
| Accelerate a big ODE                      | Use [CUDA.jl's](https://cuda.juliagpu.org/stable/) CuArray as `u0`                                       |
| Solve the same ODE with many `u0` and `p` | Use [DiffEqGPU.jl's](https://docs.sciml.ai/DiffEqGPU/stable/) `EnsembleGPUArray` and `EnsembleGPUKernel` |

## Supported GPUs

SciML's GPU support extends to a wide array of hardware, including:

| GPU Manufacturer | GPU Kernel Language | Julia Support Package                              | Backend Type             |
|:---------------- |:------------------- |:-------------------------------------------------- |:------------------------ |
| NVIDIA           | CUDA                | [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)     | `CUDA.CUDABackend()`     |
| AMD              | ROCm                | [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) | `AMDGPU.ROCBackend()`    |
| Intel            | OneAPI              | [OneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl) | `oneAPI.oneAPIBackend()` |
| Apple (M-Series) | Metal               | [Metal.jl](https://github.com/JuliaGPU/Metal.jl)   | `Metal.MetalBackend()`   |

For this tutorial we will demonstrate the CUDA backend for NVIDIA GPUs, though any of the other GPUs can be
used by simply swapping out the `backend` choice.

## Example of Within-Method GPU Parallelism

```julia
using OrdinaryDiffEq, CUDA, LinearAlgebra
u0 = cu(rand(1000))
A = cu(randn(1000, 1000))
f(du, u, p, t) = mul!(du, A, u)
prob = ODEProblem(f, u0, (0.0f0, 1.0f0)) # Float32 is better on GPUs!
sol = solve(prob, Tsit5())
```

## Example of Parameter-Parallelism with GPU Ensemble Methods

```julia
using DiffEqGPU, CUDA, OrdinaryDiffEq, StaticArrays

function lorenz(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return SVector{3}(du1, du2, du3)
end

u0 = @SVector [1.0f0; 0.0f0; 0.0f0]
tspan = (0.0f0, 10.0f0)
p = @SVector [10.0f0, 28.0f0, 8 / 3.0f0]
prob = ODEProblem{false}(lorenz, u0, tspan, p)
prob_func = (prob, i, repeat) -> remake(prob, p = (@SVector rand(Float32, 3)) .* p)
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

@time sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend()),
    trajectories = 10_000, adaptive = false, dt = 0.1f0)
```

## Benchmarks

Curious about our claims? See [https://github.com/utkarsh530/GPUODEBenchmarks](https://github.com/utkarsh530/GPUODEBenchmarks) for comparsion of our GPU solvers against CPUs and GPUs implementation in C++, JAX and PyTorch.

## Citation

If you are using `DiffEqGPU.jl` in your work, consider citing our paper:

```
@article{utkarsh2024automated,
  title={Automated translation and accelerated solving of differential equations on multiple GPU platforms},
  author={Utkarsh, Utkarsh and Churavy, Valentin and Ma, Yingbo and Besard, Tim and Srisuma, Prakitr and Gymnich, Tim and Gerlach, Adam R and Edelman, Alan and Barbastathis, George and Braatz, Richard D and others},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={419},
  pages={116591},
  year={2024},
  publisher={Elsevier}
}
```
