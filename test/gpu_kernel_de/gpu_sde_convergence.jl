using DiffEqGPU, OrdinaryDiffEq, StaticArrays, LinearAlgebra, Statistics
using DiffEqDevTools

include("../utils.jl")

u₀ = SA[0.1f0]
f(u, p, t) = SA[p[1] * u[1]]
g(u, p, t) = SA[p[2] * u[1]]
tspan = (0.0f0, 1.0f0)
p = SA[1.5f0, 0.01f0]

prob = SDEProblem(f, g, u₀, tspan, p; seed = 1234)

dts = 1 .// 2 .^ (5:-1:2)

ensemble_prob = EnsembleProblem(prob;
                                output_func = (sol, i) -> (sol[end], false))

@info "EM"
dts = 1 .// 2 .^ (12:-1:8)
sim = test_convergence(Float32.(dts), ensemble_prob, GPUEM(),
                       EnsembleGPUKernel(device, 0.0),
                       save_everystep = false, trajectories = Int(1e5),
                       weak_timeseries_errors = false,
                       expected_value = SA[u₀ * exp((p[1]))])

@show sim.𝒪est[:weak_final]
@test abs(sim.𝒪est[:weak_final] - 1.0) < 0.1

@info "GPUSIEA"

dts = 1 .// 2 .^ (6:-1:4)

sim = test_convergence(Float32.(dts), ensemble_prob, GPUSIEA(),
                       EnsembleGPUKernel(device, 0.0),
                       save_everystep = false, trajectories = Int(5e4),
                       expected_value = SA[u₀ * exp((p[1]))])

@show sim.𝒪est[:weak_final]
@test abs(sim.𝒪est[:weak_final] - 2.1) < 0.2
