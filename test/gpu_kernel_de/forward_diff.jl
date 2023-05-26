
using DiffEqGPU, OrdinaryDiffEq, StaticArrays, LinearAlgebra
include("../utils.jl")

using ForwardDiff

function lorenz(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return SVector{3}(du1, du2, du3)
end

tspan = (0.0f0, 10.0f0)

u0 = [1.0f0, 0.0f0, 0.0f0]
p = [10.0f0, 28.0f0, 8 / 3.0f0]

function ode_solve(x, alg)
    u0 = SVector{3}(x[1], x[2], x[3])
    p = SVector{3}(x[4], x[5], x[6])
    prob = ODEProblem{false}(lorenz, u0, tspan, p)

    prob_func = (prob, i, repeat) -> remake(prob, p = (@SVector rand(3)) .* p)
    monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

    sol = solve(monteprob, alg, EnsembleGPUKernel(CUDA.CUDABackend(), 0.0),
                trajectories = 2, adaptive = true, dt = 0.01f0)

    Array(sol)
end

for alg in (GPUTsit5(), GPUVern7(), GPUVern9())
    @info alg
    ForwardDiff.jacobian(x -> ode_solve(x, alg),
                         [1.0f0, 0.0f0, 0.0f0, 10.0f0, 28.0f0, 8 / 3.0f0])
end
