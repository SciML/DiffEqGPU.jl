# [Massively Parallel ODE Solving using lower API for lower overheads](@id events)


There are overheads in GPU solves when using `EnsembleGPUKernel` (For eg. offloading GPU Arrays to CPU). The example below provides a way to generate solves using lower level API with lower overheads:

```julia
using DiffEqGPU, StaticArrays, CUDA

trajectories =  10_000 

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
prob_func = (prob, i, repeat) -> remake(prob, p = (@SVector rand(Float32, 3)).*p)
ensembleProb = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

## Building different problems for different parameters
I = 1:trajectories
if ensembleProb.safetycopy
    probs = map(I) do i
        ensembleProb.prob_func(deepcopy(ensembleProb.prob), i, 1)
    end
else
    probs = map(I) do i
        ensembleProb.prob_func(ensembleProb.prob, i, 1)
    end
end

## Make them compatible with CUDA
probs = cu(probs)

## Finally use the lower API for faster solves! (Fixed time-stepping)
@time ts,us = DiffEqGPU.vectorized_solve(probs, ensembleProb.prob, GPUTsit5(); save_everystep = false, dt = 0.1f0)

## Adaptive time-stepping
@time ts,us = DiffEqGPU.vectorized_asolve(probs, ensembleProb.prob, GPUTsit5(); save_everystep = false, dt = 0.1f0)

```
