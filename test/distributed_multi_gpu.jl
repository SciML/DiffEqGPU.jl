using Distributed
addprocs(2)
@everywhere using DiffEqGPU, OrdinaryDiffEq, Test, Random
@everywhere include("utils.jl")

@everywhere begin
    function lorenz_distributed(du, u, p, t)
        du[1] = p[1] * (u[2] - u[1])
        du[2] = u[1] * (p[2] - u[3]) - u[2]
        du[3] = u[1] * u[2] - p[3] * u[3]
    end
    u0 = Float32[1.0; 0.0; 0.0]
    tspan = (0.0f0, 100.0f0)
    p = (10.0f0, 28.0f0, 8 / 3.0f0)
    Random.seed!(1)
    pre_p_distributed = [rand(Float32, 3) for i in 1:10]
    function prob_func_distributed(prob, i, repeat)
        remake(prob, p = pre_p_distributed[i] .* p)
    end
end

prob = ODEProblem(lorenz_distributed, u0, tspan, p)
monteprob = EnsembleProblem(prob, prob_func = prob_func_distributed)

#Performance check with nvvp
# CUDAnative.CUDAdrv.@profile
@time sol = solve(
    monteprob, Tsit5(), EnsembleGPUArray(backend), trajectories = 10,
    saveat = 1.0f0
)
@test length(filter(x -> x.u != sol.u[1].u, sol.u)) != 0 # 0 element array
@time sol = solve(
    monteprob, ROCK4(), EnsembleGPUArray(backend), trajectories = 10,
    saveat = 1.0f0
)
@time sol2 = solve(
    monteprob, Tsit5(), EnsembleGPUArray(backend), trajectories = 10,
    batch_size = 5, saveat = 1.0f0
)

@test length(filter(x -> x.u != sol.u[1].u, sol.u)) != 0 # 0 element array
@test length(filter(x -> x.u != sol2.u[6].u, sol.u)) != 0 # 0 element array
@test all(all(sol.u[i].prob.p .== pre_p_distributed[i] .* p) for i in 1:10)
@test all(all(sol2.u[i].prob.p .== pre_p_distributed[i] .* p) for i in 1:10)

#To set 1 GPU per device:
#=
using Distributed
addprocs(numgpus)
import CUDAdrv, CUDAnative

let gpuworkers = asyncmap(collect(zip(workers(), CUDAdrv.devices()))) do (p, d)
  remotecall_wait(CUDAnative.device!, p, d)
  p
end
=#
