# ode checks
using OrdinaryDiffEq, DiffEqGPU, Test

seed = 100
using Random;
Random.seed!(seed);
ra = rand(100)

function f!(du, u, p, t)
    du[1] = 1.01 * u[1]
end

prob = ODEProblem(f!, [0.5], (0.0, 1.0))

function output_func(sol, i)
    last(sol), false
end

function prob_func(prob, i, repeat)
    remake(prob, u0 = ra[i] * prob.u0)
end

function reduction(u, batch, I)
    u .+ sum(batch), false
end

# no reduction
prob1 = EnsembleProblem(prob, prob_func = prob_func, output_func = output_func)
sim1 = @time solve(prob1, Tsit5(), trajectories = 100, batch_size = 20)

# reduction and EnsembleThreads()
prob2 = EnsembleProblem(
    prob,
    prob_func = prob_func,
    output_func = output_func,
    reduction = reduction,
    u_init = Vector{eltype(prob.u0)}([0.0]),
)
sim2 = @time solve(prob2, Tsit5(), trajectories = 100, batch_size = 20)


# EnsembleCPUArray() and EnsembleGPUArray()
sim3 = @time solve(prob2, Tsit5(), EnsembleCPUArray(), trajectories = 100, batch_size = 20)
sim4 = @time solve(prob2, Tsit5(), EnsembleGPUArray(), trajectories = 100, batch_size = 20)

@info sim2[1]

@test sum(sim1.u) ≈ sim2.u
@test sim2.u ≈ sim3.u
@test sim2.u ≈ sim4.u
