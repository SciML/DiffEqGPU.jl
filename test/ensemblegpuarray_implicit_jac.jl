using DiffEqGPU, Test
using OrdinaryDiffEq: Rosenbrock23, Rodas5P, Tsit5
using SciMLBase: EnsembleProblem, EnsembleSerial, ODEFunction, ODEProblem, remake, solve

include("utils.jl")

# Implicit methods in `EnsembleGPUArray` factorize one Jacobian block per trajectory, built
# from `f.jac`. Without `f.jac` there is nothing to factorize, so the solve must be rejected
# instead of running the batched linear solve on an unfactorized matrix.

function lorenz!(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
    return nothing
end

function lorenz_jac!(J, u, p, t)
    J[1, 1] = -p[1]
    J[1, 2] = p[1]
    J[1, 3] = 0
    J[2, 1] = p[2] - u[3]
    J[2, 2] = -1
    J[2, 3] = -u[1]
    J[3, 1] = u[2]
    J[3, 2] = u[1]
    J[3, 3] = -p[3]
    return nothing
end

const u0 = Float32[1, 0, 0]
const tspan = (0.0f0, 1.0f0)
const p = Float32[10, 28, 8 / 3]
prob_func(prob, ctx) = remake(prob; p = prob.p .* (1 + ctx.sim_id / 10))
const kw = (; trajectories = 3, save_everystep = false, abstol = 1.0f-6, reltol = 1.0f-3)
endpoints(sol) = [Array(s.u[end]) for s in sol.u]

@testset "Implicit EnsembleGPUArray without f.jac ($(nameof(typeof(alg))))" for alg in (
        Rosenbrock23(), Rodas5P(),
    )
    ens = EnsembleProblem(ODEProblem(lorenz!, u0, tspan, p); prob_func)
    @test_throws ArgumentError solve(ens, alg, EnsembleGPUArray(backend); kw...)
end

@testset "Implicit EnsembleGPUArray with f.jac ($(nameof(typeof(alg))))" for alg in (
        Rosenbrock23(), Rodas5P(),
    )
    f = ODEFunction(lorenz!; jac = lorenz_jac!)
    ens = EnsembleProblem(ODEProblem(f, u0, tspan, p); prob_func)
    ref = endpoints(solve(ens, alg, EnsembleSerial(); kw...))
    sol = endpoints(solve(ens, alg, EnsembleGPUArray(backend); kw...))
    @test all(isapprox.(sol, ref; rtol = 1.0e-2))
end

@testset "Explicit EnsembleGPUArray does not need f.jac" begin
    ens = EnsembleProblem(ODEProblem(lorenz!, u0, tspan, p); prob_func)
    ref = endpoints(solve(ens, Tsit5(), EnsembleSerial(); kw...))
    sol = endpoints(solve(ens, Tsit5(), EnsembleGPUArray(backend); kw...))
    @test all(isapprox.(sol, ref; rtol = 1.0e-2))
end
