# Getting Started with GPU-Accelerated Differential Equations in Julia

## The two ways to accelerate ODE solvers with GPUs

There are two very different ways that one can
accelerate an ODE solution with GPUs. There is one case where `u` is very big and `f`
is very expensive but very structured, and you use GPUs to accelerate the computation
of said `f`. The other use case is where `u` is very small, but you want to solve the ODE
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

## Simple Example of Within-Method GPU Parallelism

The following is a quick and dirty example of doing within-method GPU parallelism.
Let's say we had a simple but large ODE with many linear algebra or map/broadcast
operations:

```@example basic
using OrdinaryDiffEq, LinearAlgebra
u0 = rand(1000)
A = randn(1000, 1000)
f(du, u, p, t) = mul!(du, A, u)
prob = ODEProblem(f, u0, (0.0, 1.0))
sol = solve(prob, Tsit5())
```

Translating this to a GPU-based solve of the ODE simply requires moving the arrays for
the initial condition, parameters, and caches to the GPU. This looks like:

```@example basic
using OrdinaryDiffEq, CUDA, LinearAlgebra
u0 = cu(rand(1000))
A = cu(randn(1000, 1000))
f(du, u, p, t) = mul!(du, A, u)
prob = ODEProblem(f, u0, (0.0f0, 1.0f0)) # Float32 is better on GPUs!
sol = solve(prob, Tsit5())
```

Notice that the solution values `sol[i]` are CUDA-based arrays, which can be moved back
to the CPU using `Array(sol[i])`.

More details on effective use of within-method GPU parallelism can be found in
[the within-method GPU parallelism tutorial](@ref withingpu).

## Example of Parameter-Parallelism with GPU Ensemble Methods

On the other side of the spectrum, what if we want to solve tons of small ODEs? For this
use case, we would use the ensemble methods to solve the same ODE many times with
different parameters. This looks like:

```@example basic
using DiffEqGPU, OrdinaryDiffEq, StaticArrays, CUDA

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

sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend()),
    trajectories = 10_000)
```
Another example is [GPU Ensemble Simulation with Random Decay Rates](tutorials/gpu_ensemble_random_decay.md) which showcases uncertainity quantificaiton. 

To dig more into this example, see the [ensemble GPU solving tutorial](@ref lorenz).
