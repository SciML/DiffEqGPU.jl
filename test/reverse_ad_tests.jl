using OrdinaryDiffEq, Optimization, OptimizationOptimisers, DiffEqGPU, Test
import Zygote

include("utils.jl")

function modelf(du, u, p, t)
    return du[1] = 1.01 * u[1] * p[1] * p[2]
end

function model(θ, ensemblealg)
    prob = ODEProblem(modelf, [θ[1]], (0.0, 1.0), [θ[2], θ[3]])

    function prob_func(prob, i, repeat)
        return remake(prob, u0 = 0.5 .+ i / 100 .* prob.u0)
    end

    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
    return solve(
        ensemble_prob, Tsit5(), ensemblealg, saveat = 0.1,
        trajectories = 10
    )
end

callback = function (θ, l) # callback function to observe training
    @show l
    return false
end

pa = [1.0, 2.0]
u0 = [3.0]

θ = [u0; pa]

opt = Adam(0.1)
loss_gpu(θ) = sum(abs2, 1.0 .- Array(model(θ, EnsembleCPUArray())))
l1 = loss_gpu(θ)

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_gpu(x), adtype)
optprob = Optimization.OptimizationProblem(optf, θ)

res_gpu = Optimization.solve(optprob, opt; callback = callback, maxiters = 100)
