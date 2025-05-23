"""
```julia
EnsembleCPUArray(cpu_offload = 0.2)
```

An `EnsembleArrayAlgorithm` which utilizes the CPU kernels to parallelize each ODE solve
with their separate ODE integrator on each kernel. This method is meant to be a debugging
counterpart to `EnsembleGPUArray`, having the same behavior and using the same
KernelAbstractions.jl process to build the combined ODE, but without the restrictions of
`f` being a GPU-compatible kernel function.

It is unlikely that this method is useful beyond library development and debugging, as
almost any case should be faster with `EnsembleThreads` or `EnsembleDistributed`.
"""
struct EnsembleCPUArray <: EnsembleArrayAlgorithm end

"""
```julia
EnsembleGPUArray(backend, cpu_offload = 0.2)
```

An `EnsembleArrayAlgorithm` which utilizes the GPU kernels to parallelize each ODE solve
with their separate ODE integrator on each kernel.

## Positional Arguments

  - `backend`: the KernelAbstractions backend for performing the computation.
  - `cpu_offload`: the percentage of trajectories to offload to the CPU. Default is 0.2 or
    20% of trajectories.

## Limitations

`EnsembleGPUArray` requires being able to generate a kernel for `f` using
KernelAbstractons.jl and solving the resulting ODE defined over `CuArray` input types.
This introduces the following limitations on its usage:

  - Not all standard Julia `f` functions are allowed. Only Julia `f` functions which are
    capable of being compiled into a GPU kernel are allowed. This notably means that
    certain features of Julia can cause issues inside of kernel, like:

      + Allocating memory (building arrays)
      + Linear algebra (anything that calls BLAS)
      + Broadcast

  - Not all ODE solvers are allowed, only those from OrdinaryDiffEq.jl. The tested feature
    set from OrdinaryDiffEq.jl includes:

      + Explicit Runge-Kutta methods
      + Implicit Runge-Kutta methods
      + Rosenbrock methods
      + DiscreteCallbacks and ContinuousCallbacks
  - Stiff ODEs require the analytical solution of every derivative function it requires.
    For example, Rosenbrock methods require the Jacobian and the gradient with respect to
    time, and so these two functions are required to be given. Note that they can be
    generated by the
    [modelingtoolkitize](https://docs.juliadiffeq.org/latest/tutorials/advanced_ode_example/#Automatic-Derivation-of-Jacobian-Functions-1)
    approach.
  - To use multiple GPUs over clusters, one must manually set up one process per GPU. See the
    multi-GPU tutorial for more details.

!!! warn

    Callbacks with `terminate!` does not work well with `EnsembleGPUArray` as the entire
    integration will hault when any of the trajectories hault. Use with caution.

## Example

```julia
using DiffEqGPU, CUDA, OrdinaryDiffEq
function lorenz(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

u0 = Float32[1.0; 0.0; 0.0]
tspan = (0.0f0, 100.0f0)
p = [10.0f0, 28.0f0, 8 / 3.0f0]
prob = ODEProblem(lorenz, u0, tspan, p)
prob_func = (prob, i, repeat) -> remake(prob, p = rand(Float32, 3) .* p)
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)
@time sol = solve(monteprob, Tsit5(), EnsembleGPUArray(CUDADevice()),
    trajectories = 10_000, saveat = 1.0f0)
```
"""
struct EnsembleGPUArray{Backend} <: EnsembleArrayAlgorithm
    backend::Backend
    cpu_offload::Float64
end

"""
```julia
EnsembleGPUKernel(backend, cpu_offload = 0.0)
```

A massively-parallel ensemble algorithm which generates a unique GPU kernel for the entire
ODE which is then executed. This leads to a very low overhead GPU code generation, but
imparts some extra limitations on the use.

## Positional Arguments

  - `backend`: the KernelAbstractions backend for performing the computation.
  - `cpu_offload`: the percentage of trajectories to offload to the CPU. Default is 0.0 or
    0% of trajectories.

## Limitations

  - Not all standard Julia `f` functions are allowed. Only Julia `f` functions which are
    capable of being compiled into a GPU kernel are allowed. This notably means that
    certain features of Julia can cause issues inside a kernel, like:

      + Allocating memory (building arrays)
      + Linear algebra (anything that calls BLAS)
      + Broadcast

  - Only out-of-place `f` definitions are allowed. Coupled with the requirement of not
    allowing for memory allocations, this means that the ODE must be defined with
    `StaticArray` initial conditions.
  - Only specific ODE solvers are allowed. This includes:

      + GPUTsit5
      + GPUVern7
      + GPUVern9
  - To use multiple GPUs over clusters, one must manually set up one process per GPU. See the
    multi-GPU tutorial for more details.

## Example

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

@time sol = solve(
    monteprob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend()), trajectories = 10_000,
    adaptive = false, dt = 0.1f0)
```
"""
struct EnsembleGPUKernel{Dev} <: EnsembleKernelAlgorithm
    dev::Dev
    cpu_offload::Float64
end

cpu_alg = Dict(GPUTsit5 => (GPUSimpleTsit5(), GPUSimpleATsit5()),
    GPUVern7 => (GPUSimpleVern7(), GPUSimpleAVern7()),
    GPUVern9 => (GPUSimpleVern9(), GPUSimpleAVern9()))

# Work around the fact that Zygote cannot handle the task system
# Work around the fact that Zygote isderiving fails with constants?
function EnsembleGPUArray(dev)
    EnsembleGPUArray(dev, 0.2)
end

function EnsembleGPUKernel(dev)
    EnsembleGPUKernel(dev, 0.0)
end

function ChainRulesCore.rrule(::Type{<:EnsembleGPUArray})
    EnsembleGPUArray(0.0), _ -> NoTangent()
end

ZygoteRules.@adjoint function EnsembleGPUArray(dev)
    EnsembleGPUArray(dev, 0.0), _ -> nothing
end
