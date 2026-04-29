# Using GPU-accelerated Ensembles with Automatic Differentiation

`EnsembleGPUArray` comes with derivative overloads for reverse mode automatic differentiation,
and thus can be thrown into deep learning training loops. The following is an example
of this use:

```julia
using OrdinaryDiffEq, SciMLSensitivity, Flux, DiffEqGPU, CUDA

CUDA.allowscalar(false)

pa = [1.0, 2.0]
u0 = [3.0]

function modelf(du, u, p, t)
    du[1] = 1.01 * u[1] * p[1] * p[2]
end

function model(p)
    prob = ODEProblem(modelf, u0, (0.0, 1.0), p)

    function prob_func(prob, ctx)
        remake(prob, u0 = 0.5 .+ ctx.sim_id / 100 .* prob.u0)
    end

    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
    solve(ensemble_prob, Tsit5(), EnsembleGPUArray(CUDA.CUDABackend()), saveat = 0.1,
        trajectories = 10)
end

# loss function
loss(p) = sum(abs2, 1.0 .- Array(model(p)))

println("Starting to train")

l1 = loss(pa)
@show l1

# Use Flux's gradient descent with explicit parameter updates
opt_state = Flux.setup(Adam(0.1), pa)
for epoch in 1:10
    grads = Flux.gradient(loss, pa)
    Flux.update!(opt_state, pa, grads[1])
    @show loss(pa)
end
```

Forward-mode automatic differentiation works as well, as demonstrated by its capability
to recompile for Dual number arithmetic:

```@example ad
using OrdinaryDiffEq, DiffEqGPU, ForwardDiff, Test

function lorenz(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

u0 = [ForwardDiff.Dual(1.0f0, (1.0, 0.0, 0.0)), ForwardDiff.Dual(0.0f0, (0.0, 1.0, 0.0)),
    ForwardDiff.Dual(0.0f0, (0.0, 0.0, 1.0))]
tspan = (0.0f0, 100.0f0)
p = (10.0f0, 28.0f0, 8 / 3.0f0)
prob = ODEProblem{true, SciMLBase.FullSpecialize}(lorenz, u0, tspan, p)
prob_func = (prob, ctx) -> remake(prob, p = rand(Float32, 3) .* p)
monteprob = EnsembleProblem(prob, prob_func = prob_func)
@time sol = solve(monteprob, Tsit5(), EnsembleGPUArray(CUDA.CUDABackend()),
    trajectories = 10_000,
    saveat = 1.0f0)
```
