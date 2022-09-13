using DiffEqGPU, SimpleDiffEq, StaticArrays, CUDA, BenchmarkTools, OrdinaryDiffEq
using Plots

CUDA.allowscalar(false)

function f(u, p, t)
    du1 = -u[1]
    return SVector{1}(du1)
end

u0 = @SVector [10.0f0]
prob = ODEProblem{false}(f, u0, (0.0f0, 10.0f0))
prob_func = (prob, i, repeat) -> remake(prob, p = prob.p)
monteprob = EnsembleProblem(prob, safetycopy = false)
const V = 1

condition(u, t, integrator) = t == 4.0f0
affect!(integrator) = integrator.u += @SVector[10.0f0]
cb = GPUDiscreteCallback(condition, affect!)

sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(),
            trajectories = 2,
            adaptive = false, dt = 0.1f0
            tstops = CuArray([4.0f0]))

#plot(sol[1])
