using DiffEqGPU, OrdinaryDiffEq, StaticArrays, LinearAlgebra
function lorenz(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return SVector{3}(du1, du2, du3)
end

u0 = @SVector [1.0f0; 0.0f0; 0.0f0]
tspan = (0.0f0, 10.0f0)
p = @SVector [10.0f0, 28.0f0, 8 / 3.0f0]
prob = ODEProblem{false}(lorenz, u0, tspan, p)
prob_func = (prob, i, repeat) -> remake(prob, p = p)
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), trajectories = 10,
            adaptive = false, dt = 0.01f0)
asol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), trajectories = 10,
             adaptive = true, dt = 0.1f-1, abstol = 1.0f-8, reltol = 1.0f-5)

@test sol.converged == true
@test asol.converged == true

## Regression test

bench_sol = solve(prob, Tsit5(), adaptive = false, dt = 0.01f0)
bench_asol = solve(prob, Tsit5(), dt = 0.1f-1, save_everystep = false, abstol = 1e-8,
                   reltol = 1e-5)

@test norm(bench_sol.u[end] - sol[1].u[end]) < 5e-3
@test norm(bench_asol.u - asol[1].u) < 1e-4

### solve parameters

saveat = [0.0f0, 4.0f0]

sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), trajectories = 2,
            adaptive = false, dt = 0.01f0, saveat = saveat)

asol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), trajectories = 2,
             adaptive = true, dt = 0.1f-1, abstol = 1.0f-8, reltol = 1.0f-5,
             saveat = saveat)

bench_sol = solve(prob, Tsit5(), adaptive = false, dt = 0.01f0, saveat = saveat)
bench_asol = solve(prob, Tsit5(), dt = 0.1f-1, save_everystep = false, abstol = 1e-8,
                   reltol = 1e-5, saveat = saveat)

@test norm(bench_sol.u - sol[1].u) < 1e-4
@test norm(bench_asol.u - asol[1].u) < 1e-4

@test length(sol[1].u) == length(saveat)
@test length(asol[1].u) == length(saveat)

sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), trajectories = 2,
            adaptive = false, dt = 0.01f0, save_everystep = false)

bench_sol = solve(prob, Tsit5(), adaptive = false, dt = 0.01f0, save_everystep = false)

@test norm(bench_sol.u - sol[1].u) < 5e-3

@test length(sol[1].u) == length(asol[1].u)

## With random parameters

prob_func = (prob, i, repeat) -> remake(prob, p = (@SVector rand(Float32, 3)) .* p)
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), trajectories = 10,
            adaptive = false, dt = 0.1f0)
asol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), trajectories = 10,
             adaptive = true, dt = 0.1f-1, abstol = 1.0f-8, reltol = 1.0f-5)
