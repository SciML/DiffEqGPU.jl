using DiffEqGPU, OrdinaryDiffEq, StaticArrays, LinearAlgebra, CUDA, Statistics

u₀ = SA[0.1f0]
f(u, p, t) = SA[p[1] * u[1]]
g(u, p, t) = SA[p[2] * u[1]]
tspan = (0.0f0, 1.0f0)
p = SA[1.5f0, 0.01f0]

prob = SDEProblem(f, g, u₀, tspan, p; seed = 1234)

## Testing convergence using Linear Regression
function test_convergence(prob, alg, dts; numtraj = Int(5e4), kwargs...)
    monteprob = EnsembleProblem(prob)

    errs = Float64[]

    for dt in dts
        sol = solve(monteprob, alg, EnsembleGPUKernel(0.0), dt = Float32(dt),
                    trajectories = numtraj; kwargs...)
        sol_array = Array(sol)

        us = reshape(mean(sol_array, dims = 3), size(sol_array, 2))

        us_exact = u₀ .* exp.(p[1] * sol[1].t)

        push!(errs, norm(us - us_exact, Inf))
    end

    A = [ones(length(dts)) log.(dts)]

    b = log.(errs)

    coeffs = A \ b

    return coeffs[2]
end

dts = 1 .// 2 .^ (12:-1:8)
em_weak_order = test_convergence(prob, GPUEM(), dts, adaptive = false,
                                 save_everystep = false)

@test isapprox(em_weak_order, 1.011; atol = 1e-3)

dts = 1 .// 2 .^ (6:-1:3)
siea_weak_order = test_convergence(prob, GPUSIEA(), dts, adaptive = false,
                                   save_everystep = false)

@test isapprox(siea_weak_order, 2.024; atol = 1e-3)
