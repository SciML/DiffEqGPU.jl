# GPU Parallel Solving of Stochastic Differential Equations

One major application of DiffEqGPU is for computing ensemble statistics of SDE solutions
using `EnsembleGPUArray`. The following demonstrates using this technique to generate
large ensembles of solutions for a diagonal noise SDE with a high order adaptive method:

```@example sde
using DiffEqGPU, CUDA, StochasticDiffEq

function lorenz(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

function multiplicative_noise(du, u, p, t)
    du[1] = 0.1 * u[1]
    du[2] = 0.1 * u[2]
    du[3] = 0.1 * u[3]
end

CUDA.allowscalar(false)
u0 = Float32[1.0; 0.0; 0.0]
tspan = (0.0f0, 10.0f0)
p = (10.0f0, 28.0f0, 8 / 3.0f0)
prob = SDEProblem(lorenz, multiplicative_noise, u0, tspan, p)
const pre_p = [rand(Float32, 3) for i in 1:10]
prob_func = (prob, i, repeat) -> remake(prob, p = pre_p[i] .* p)
monteprob = EnsembleProblem(prob, prob_func = prob_func)
sol = solve(monteprob, SOSRI(), EnsembleGPUArray(CUDA.CUDABackend()), trajectories = 10_000, saveat = 1.0f0)
```
