# Batched Reductions for Lowering Peak Memory Requirements

Just as in the regular form of the
[DifferentialEquations.jl ensemble interface](https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/),
a `reduction` function can be given to reduce between batches. Here we show an example
of running 20 ODEs at a time, grabbing its value at the end, and reducing by summing all
the values. This then allows for only saving the sum of the previous batches, boosting
the trajectory count to an amount that is higher than would fit in memory, and only saving
the summed values.

```@example reductions
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

prob2 = EnsembleProblem(prob, prob_func = prob_func, output_func = output_func,
                        reduction = reduction, u_init = Vector{eltype(prob.u0)}([0.0]))
sim4 = solve(prob2, Tsit5(), EnsembleGPUArray(), trajectories = 100, batch_size = 20)
```
