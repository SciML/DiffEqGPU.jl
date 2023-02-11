using OrdinaryDiffEq, SciMLSensitivity, Flux, DiffEqGPU, CUDA, CUDAKernels, Test
CUDA.allowscalar(false)

function modelf(du, u, p, t)
    du[1] = 1.01 * u[1] * p[1] * p[2]
end

function model()
    prob = ODEProblem(modelf, u0, (0.0, 1.0), pa)

    function prob_func(prob, i, repeat)
        remake(prob, u0 = 0.5 .+ i / 100 .* prob.u0)
    end

    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
    solve(ensemble_prob, Tsit5(), EnsembleGPUArray(CUDADevice()), saveat = 0.1,
          trajectories = 10)
end

# loss function
loss() = sum(abs2, 1.0 .- Array(model()))

data = Iterators.repeated((), 10)

cb = function () # callback function to observe training
    @show loss()
end

pa = [1.0, 2.0]
u0 = [3.0]
opt = ADAM(0.1)
println("Starting to train")

l1 = loss()

Flux.@epochs 10 Flux.train!(loss, Flux.params([pa]), data, opt; cb = cb)

l2 = loss()
@test 3l2 < l1
