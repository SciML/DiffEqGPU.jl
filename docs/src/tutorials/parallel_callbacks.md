# [Massively Parallel ODE Solving with Event Handling and Callbacks](@id events)

```julia
using DiffEqGPU, StaticArrays, OrdinaryDiffEq
function f(u, p, t)
    du1 = -u[1]
    return SVector{1}(du1)
end

u0 = @SVector [10.0f0]
prob = ODEProblem{false}(f, u0, (0.0f0, 10.0f0))
prob_func = (prob, i, repeat) -> remake(prob, p = prob.p)
monteprob = EnsembleProblem(prob, safetycopy = false)

condition(u, t, integrator) = t == 4.0f0
affect!(integrator) = integrator.u += @SVector[10.0f0]

gpu_cb = DiscreteCallback(condition, affect!; save_positions = (false, false))

sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(),
            trajectories = 10,
            adaptive = false, dt = 0.01f0, callback = gpu_cb, merge_callbacks = true,
            tstops = [4.0f0])
```
