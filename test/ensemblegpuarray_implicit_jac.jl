using DiffEqGPU, Test
using OrdinaryDiffEq: Rodas5P, Rosenbrock23, Tsit5
using OrdinaryDiffEqSDIRK: ImplicitEuler
using OrdinaryDiffEqNonlinearSolve: NLFunctional
using SciMLBase: EnsembleProblem, EnsembleSerial, ODEFunction, ODEProblem, ReturnCode,
    remake, solve

include("utils.jl")

# Implicit methods in `EnsembleGPUArray` factorize one Jacobian block per trajectory when
# `f.jac` is given and use the algorithm's own linear solver otherwise. Both must agree with
# solving each trajectory separately.

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

cases = (
    ("Rosenbrock23 without jac", Rosenbrock23(), lorenz!, (;)),
    ("Rodas5P without jac", Rodas5P(), lorenz!, (;)),
    ("Rosenbrock23 with jac", Rosenbrock23(), ODEFunction(lorenz!; jac = lorenz_jac!), (;)),
    ("Rodas5P with jac", Rodas5P(), ODEFunction(lorenz!; jac = lorenz_jac!), (;)),
    (
        "ImplicitEuler with NLFunctional", ImplicitEuler(nlsolve = NLFunctional()), lorenz!,
        (; adaptive = false, dt = 1.0f-3),
    ),
    ("Tsit5", Tsit5(), lorenz!, (;)),
)

@testset "EnsembleGPUArray $name matches EnsembleSerial" for (name, alg, f, extra) in cases
    ens = EnsembleProblem(ODEProblem(f, u0, tspan, p); prob_func)
    kw = (;
        trajectories = 3, save_everystep = false, abstol = 1.0f-6, reltol = 1.0f-3, extra...,
    )
    ref = solve(ens, alg, EnsembleSerial(); kw...)
    sol = solve(ens, alg, EnsembleGPUArray(backend); kw...)
    for i in 1:3
        @test sol.u[i].retcode == ReturnCode.Success
        @test sol.u[i].t[end] == tspan[2]
        @test isapprox(Array(sol.u[i].u[end]), ref.u[i].u[end]; rtol = 1.0e-2)
    end
end
