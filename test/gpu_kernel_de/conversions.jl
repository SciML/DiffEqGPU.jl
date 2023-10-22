using DiffEqGPU, OrdinaryDiffEq, StaticArrays, LinearAlgebra, Test
include("../utils.jl")

function lorenz(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return SVector{3}(du1, du2, du3)
end

u0 = [1.0f0; 0.0f0; 0.0f0]
tspan = (0.0f0, 10.0f0)
p = [10.0f0, 28.0f0, 8 / 3.0f0]
prob = ODEProblem{false}(lorenz, u0, tspan, p)
prob_func = (prob, i, repeat) -> remake(prob, p = (@SVector rand(Float32, 3)) .* p)
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

@test solve(monteprob, GPUTsit5(), EnsembleGPUKernel(backend),
    trajectories = 10_000,
    saveat = 1.0)[1].t == 0.0f0:1.0f0:10.0f0

@test solve(monteprob, GPUTsit5(), EnsembleGPUKernel(backend),
    trajectories = 10_000,
    saveat = [1.0f0, 5.0f0, 10.0f0])[1].t == [1.0f0, 5.0f0, 10.0f0]

@test solve(monteprob, GPUTsit5(), EnsembleGPUKernel(backend),
    trajectories = 10_000,
    saveat = [1.0, 5.0, 10.0])[1].t == [1.0f0, 5.0f0, 10.0f0]

@test solve(monteprob, GPUTsit5(), EnsembleGPUKernel(backend),
    trajectories = 10_000,
    saveat = 1:10)[1].t == Float32.(1:10)

@test solve(monteprob, GPUTsit5(), EnsembleGPUKernel(backend),
    trajectories = 10_000,
    saveat = 1:0.1:10)[1].t == StepRangeLen{Float32, Float32, Float32, Int32}(1.0f0, 0.1f0, 91)

@test solve(monteprob, GPUTsit5(), EnsembleGPUKernel(backend),
    trajectories = 10_000,
    saveat = 1:(1.0f0):10)[1].t == 1:1.0f0:10
