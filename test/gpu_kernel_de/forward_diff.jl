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

u0 = @SVector [
    ForwardDiff.Dual(1.0f0, (1.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0));
    ForwardDiff.Dual(0.0f0, (0.0f0, 1.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0));
    ForwardDiff.Dual(0.0f0, (0.0f0, 0.0f0, 1.0f0, 0.0f0, 0.0f0, 0.0f0))
]

p = @SVector [
    ForwardDiff.Dual(10.0f0, (0.0f0, 0.0f0, 0.0f0, 1.0f0, 0.0f0, 0.0f0)),
    ForwardDiff.Dual(28.0f0, (0.0f0, 0.0f0, 0.0f0, 0.0f0, 1.0f0, 0.0f0)),
    ForwardDiff.Dual(8 / 3.0f0, (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 1.0f0)),
]

tspan = (0.0f0, 10.0f0)

prob = ODEProblem{false}(lorenz, u0, tspan, p)

prob_func = (prob, i, repeat) -> remake(prob, p = p)
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

for alg in (
        GPUTsit5(), GPUVern7(), GPUVern9(), GPURosenbrock23(autodiff = false),
        GPURodas4(autodiff = false), GPURodas5P(autodiff = false),
        GPUKvaerno3(autodiff = false), GPUKvaerno5(autodiff = false),
    )
    @info alg
    sol = solve(
        monteprob, alg, EnsembleGPUKernel(backend, 0.0),
        trajectories = 2, save_everystep = false, adaptive = false, dt = 0.01f0
    )
    asol = solve(
        monteprob, alg, EnsembleGPUKernel(backend, 0.0),
        trajectories = 2, adaptive = true, dt = 0.01f0
    )
end
