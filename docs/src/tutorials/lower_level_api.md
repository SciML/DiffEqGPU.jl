# [Using the Lower Level API for Decreased Overhead with GPU acclerated Ensembles](@id lowerlevel)

`EnsembleGPUKernel` is designed to match the SciML ensemble interface in order to allow for directly
converting CPU code to GPU code without any code changes. However, this hiding of the GPU aspects
decreases the overall performance as it always transfers the problem to the GPU and the result back
to the CPU for the user. These overheads can be removed by directly using the lower level API elements
of EnsembleGPUKernel.

The example below provides a way to generate solves using the lower level API with lower overheads:

```@example lower_level
using DiffEqGPU, OrdinaryDiffEq, StaticArrays, CUDA, DiffEqBase

trajectories = 10_000

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

## Building different problems for different parameters
probs = map(1:trajectories) do i
    DiffEqGPU.make_prob_compatible(remake(prob, p = (@SVector rand(Float32, 3)) .* p))
end

## Move the arrays to the GPU
probs = cu(probs)

## Finally use the lower API for faster solves! (Fixed time-stepping)

# Run once for compilation
@time CUDA.@sync ts,
us = DiffEqGPU.vectorized_solve(probs, prob, GPUTsit5();
    save_everystep = false, dt = 0.1f0)

@time CUDA.@sync ts,
us = DiffEqGPU.vectorized_solve(probs, prob, GPUTsit5();
    save_everystep = false, dt = 0.1f0)

## Adaptive time-stepping
# Run once for compilation
@time CUDA.@sync ts,
us = DiffEqGPU.vectorized_asolve(probs, prob, GPUTsit5();
    save_everystep = false, dt = 0.1f0)

@time CUDA.@sync ts,
us = DiffEqGPU.vectorized_asolve(probs, prob, GPUTsit5();
    save_everystep = false, dt = 0.1f0)
```

Note that the core is the function `DiffEqGPU.vectorized_solve` which is the solver for the CUDA-based `probs`
which uses the manually converted problems, and returns `us` which is a vector of CuArrays for the solution.

Similarily, there exists a lower-level API for `EnsembleGPUArray` as well, primarily for benchmarking purposes. The solution
returned for state (`sol.u`) is a matrix having columns as different parameter-parallel solutions for the ensemble problem.
An example is shown below:

```@example lower_level
using DiffEqGPU, OrdinaryDiffEq, StaticArrays, CUDA, DiffEqBase

trajectories = 10_000

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

## Building different problems for different parameters
batch = 1:trajectories
probs = map(batch) do i
    remake(prob, p = (@SVector rand(Float32, 3)) .* p)
end

## Finally use the lower API for faster solves! (Fixed time-stepping)

@time CUDA.@sync sol = DiffEqGPU.vectorized_map_solve(
    probs, Tsit5(), EnsembleGPUArray(0.0),
    batch, false, dt = 0.001f0,
    save_everystep = false, dense = false)

## Adaptive time-stepping (Notice the boolean argument)
@time CUDA.@sync sol = DiffEqGPU.vectorized_map_solve(
    probs, Tsit5(), EnsembleGPUArray(0.0),
    batch, true, dt = 0.001f0,
    save_everystep = false, dense = false)
```
