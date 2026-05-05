# ode checks
using OrdinaryDiffEq, DiffEqGPU, Test
include("utils.jl")

seed = 100
using Random;
Random.seed!(seed);
ra = rand(100)

function f!(du, u, p, t)
    return du[1] = 1.01 * u[1]
end

prob = ODEProblem(f!, [0.5], (0.0, 1.0))

function output_func(sol, ctx)
    # Use `sol.u[end]` rather than `last(sol)` so each per-trajectory output
    # stays a state vector. In SciMLBase v3, `last(sol)` returns a scalar for
    # single-component ODEs, which makes `sum(sim1.u)` a scalar and breaks the
    # comparison against `sim2.u` (a 1-element vector from `u_init = [0.0]`).
    return sol.u[end], false
end

function prob_func(prob, ctx)
    return remake(prob, u0 = ra[ctx.sim_id] * prob.u0)
end

function reduction(u, batch, I)
    return u .+ sum(batch), false
end

# no reduction
prob1 = EnsembleProblem(prob, prob_func = prob_func, output_func = output_func)
sim1 = @time solve(prob1, Tsit5(), trajectories = 100, batch_size = 20)

# reduction and EnsembleThreads()
prob2 = EnsembleProblem(
    prob, prob_func = prob_func, output_func = output_func,
    reduction = reduction, u_init = Vector{eltype(prob.u0)}([0.0])
)
sim2 = @time solve(prob2, Tsit5(), trajectories = 100, batch_size = 20)

# EnsembleCPUArray() and EnsembleGPUArray()
sim3 = @time solve(prob2, Tsit5(), EnsembleCPUArray(), trajectories = 100, batch_size = 20)
sim4 = @time solve(
    prob2, Tsit5(), EnsembleGPUArray(backend), trajectories = 100,
    batch_size = 20
)

@info sim2[1]

@test sum(sim1.u) ≈ sim2.u
@test sim2.u ≈ sim3.u
@test sim2.u ≈ sim4.u
