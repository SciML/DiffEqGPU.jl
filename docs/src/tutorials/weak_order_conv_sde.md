# [Using the EnsembleGPUKernel SDE solvers for the expectation of SDEs ](@id sdeweakconv)

Solving the `SDEProblem` using weak methods on multiple trajectories helps to generate the expectation of the stochastic process.
With the lower overhead of `EnsembleGPUKernel` API, these calculations can be done in parallel on GPU, potentially being faster.

The example below provides a way to calculate the expectation time-series of a linear SDE:

```julia
using DiffEqGPU, OrdinaryDiffEq, StaticArrays, LinearAlgebra, Statistics

num_trajectories = 10_000

# Defining the Problem
# dX = pudt + qudW
u₀ = SA[0.1f0]
f(u, p, t) = SA[p[1] * u[1]]
g(u, p, t) = SA[p[2] * u[1]]
tspan = (0.0f0, 1.0f0)
p = SA[1.5f0, 0.01f0]

prob = SDEProblem(f, g, u₀, tspan, p; seed = 1234)

monteprob = EnsembleProblem(prob)

sol = solve(monteprob, GPUEM(), EnsembleGPUKernel(0.0), dt = Float32(1 // 2^8),
            trajectories = num_trajectories, adaptive = false)

sol_array = Array(sol)

ts = sol[1].t

us_calc = reshape(mean(sol_array, dims = 3), size(sol_array, 2))

us_expect = u₀ .* exp.(p[1] * ts)

using Plots
plot(ts, us_expect, lw = 5,
     xaxis = "Time (t)", yaxis = "y(t)", label = "True Expected value")

plot!(ts, us_calc, lw = 3, ls = :dash, label = "Caculated Expected value")
```
