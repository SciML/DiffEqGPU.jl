# Using GPU-accelerated Ensembles with Automatic Differentiation

`EnsembleGPUArray` comes with derivative overloads for reverse mode automatic differentiation,
and thus can be thrown into deep learning training loops. The following is an example
of this use:

```@example ad
using OrdinaryDiffEq, SciMLSensitivity, Lux, Optimisers, Zygote, DiffEqGPU, CUDA, Random

CUDA.allowscalar(false)

# A tiny Lux model whose parameters are what we train. It maps a constant
# input to the two ODE parameters.
const dense = Dense(1 => 2)
const x = Float32[1.0]
rng = Random.default_rng()
Random.seed!(rng, 0)
ps, st = Lux.setup(rng, dense)

u0 = Float32[3.0]

function modelf(du, u, p, t)
    du[1] = 1.01f0 * u[1] * p[1] * p[2]
end

function model(p)
    prob = ODEProblem(modelf, u0, (0.0f0, 1.0f0), p)

    function prob_func(prob, ctx)
        remake(prob, u0 = 0.5f0 .+ Float32(ctx.sim_id) / 100 .* prob.u0)
    end

    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
    solve(ensemble_prob, Tsit5(), EnsembleGPUArray(CUDA.CUDABackend()), saveat = 0.1f0,
        trajectories = 10)
end

# loss function: run the Lux model to produce ODE parameters, then score the ensemble
function loss(ps)
    p_vec, _ = dense(x, ps, st)
    sum(abs2, 1.0f0 .- Array(model(p_vec)))
end

println("Starting to train")

l1 = loss(ps)
@show l1

# Optimisers.jl handles parameter updates; Zygote.jl handles gradients
opt_state = Optimisers.setup(Optimisers.Adam(0.1f0), ps)
for epoch in 1:10
    grads = Zygote.gradient(loss, ps)
    Optimisers.update!(opt_state, ps, grads[1])
    @show loss(ps)
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
prob = ODEProblem(lorenz, u0, tspan, p)
prob_func = (prob, ctx) -> remake(prob, p = rand(Float32, 3) .* p)
monteprob = EnsembleProblem(prob, prob_func = prob_func)
@time sol = solve(monteprob, Tsit5(), EnsembleGPUArray(CUDA.CUDABackend()),
    trajectories = 10_000,
    saveat = 1.0f0)
```
