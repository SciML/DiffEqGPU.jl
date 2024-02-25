using DiffEqGPU, StaticArrays, OrdinaryDiffEq, LinearAlgebra

include("../../utils.jl")

function rober(u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    return @SVector [
        -k₁ * y₁ + k₃ * y₂ * y₃,
        k₁ * y₁ - k₂ * y₂^2 - k₃ * y₂ * y₃,
        y₁ + y₂ + y₃ - 1]
end
function rober_jac(u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    return @SMatrix[(k₁*-1) (y₃*k₃) (k₃*y₂)
                    k₁ (y₂ * k₂ * -2+y₃ * k₃ * -1) (k₃*y₂*-1)
                    0 (y₂*2*k₂) (0)]
end
M = @SMatrix [1.0f0 0.0f0 0.0f0
              0.0f0 1.0f0 0.0f0
              0.0f0 0.0f0 0.0f0]
ff = ODEFunction(rober, mass_matrix = M)
prob = ODEProblem(ff, @SVector([1.0f0, 0.0f0, 0.0f0]), (0.0f0, 1.0f5),
    (0.04f0, 3.0f7, 1.0f4))

monteprob = EnsembleProblem(prob, safetycopy = false)

alg = GPURosenbrock23()

bench_sol = solve(prob, Rosenbrock23(), dt = 0.1, abstol = 1.0f-5, reltol = 1.0f-5)

sol = solve(monteprob, alg, EnsembleGPUKernel(backend),
    trajectories = 2,
    dt = 0.1f0,
    adaptive = true, abstol = 1.0f-5, reltol = 1.0f-5)

@test norm(bench_sol.u[1] - sol[1].u[1]) < 8e-4
@test norm(bench_sol.u[end] - sol[1].u[end]) < 8e-4
