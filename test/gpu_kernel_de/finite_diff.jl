
using DiffEqGPU, OrdinaryDiffEq, StaticArrays, LinearAlgebra
include("../utils.jl")

using ForwardDiff

function f(u, p, t)
    du1 = -p[1] * u[1] * u[1]
    return SVector{1}(du1)
end

u0 = @SVector [10.0f0]
p = @SVector [1.0f0]
tspan = (0.0f0, 10.0f0)

prob = ODEProblem{false}(f, u0, tspan, p)

prob_func = (prob, i, repeat) -> remake(prob, p = p)
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

osol = solve(prob, Rodas5P(), dt = 0.01f0, save_everystep = false)

for alg in (GPURosenbrock23(autodiff = false), GPURodas4(autodiff = false),
            GPURodas5P(autodiff = false), GPUKvaerno3(autodiff = false),
            GPUKvaerno5(autodiff = false))
    @info alg
    sol = solve(monteprob, alg, EnsembleGPUKernel(CUDA.CUDABackend(), 0.0),
                trajectories = 2, save_everystep = false, adaptive = true, dt = 0.01f0)
    @test norm(sol[1].u - osol.u) < 2e-4

    # massive threads
    sol = solve(monteprob, alg, EnsembleGPUKernel(CUDA.CUDABackend(), 0.0),
                trajectories = 10_000, save_everystep = false, adaptive = true, dt = 0.01f0)
end
