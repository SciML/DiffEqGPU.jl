using DiffEqGPU, StaticArrays, CUDA, Adapt, OrdinaryDiffEq

include("utils.jl")

@info "Testing lower level API for EnsembleGPUKernel"

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

## Move the arrays to the GPU
gpu_probs = adapt(backend, probs)

## Finally use the lower API for faster solves! (Fixed time-stepping)

@time ts, us = DiffEqGPU.vectorized_solve(gpu_probs, prob, GPUTsit5();
                                          save_everystep = false, dt = 0.1f0)

## Adaptive time-stepping
@time ts, us = DiffEqGPU.vectorized_asolve(gpu_probs, prob, GPUTsit5();
                                           save_everystep = false, dt = 0.1f0)

@info "Testing lower level API for EnsembleGPUArray"

@time sol = DiffEqGPU.vectorized_map_solve(probs, Tsit5(), EnsembleGPUArray(backend, 0.0),
                                           batch, false, dt = 0.001f0,
                                           save_everystep = false, dense = false)

## Adaptive time-stepping (Notice the boolean argument)
@time sol = DiffEqGPU.vectorized_map_solve(probs, Tsit5(), EnsembleGPUArray(backend, 0.0),
                                           batch, true, dt = 0.001f0,
                                           save_everystep = false, dense = false)
