using DiffEqGPU, OrdinaryDiffEq, StaticArrays, LinearAlgebra, CUDA
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
            adaptive = false, dt = 0.1f0)
asol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), trajectories = 10,
             adaptive = true, dt = 0.1f-1, abstol = 1.0f-8, reltol = 1.0f-5)

@test sol.converged == true
@test asol.converged == true

## Regression test

bench_sol = solve(prob, Tsit5(), adaptive = false, dt = 0.1f0)
bench_asol = solve(prob, Tsit5(), dt = 0.1f-1, save_everystep = false, abstol = 1e-8,
                   reltol = 1e-5)

@show norm(bench_sol.u[end] - sol[1].u[end]) < 2e-2
@show norm(bench_asol.u[end] - asol[1].u[end]) < 1e-4

## With random parameters

prob_func = (prob, i, repeat) -> remake(prob, p = (@SVector rand(Float32, 3)) .* p)
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), trajectories = 10,
            adaptive = false, dt = 0.1f0)
asol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(), trajectories = 10,
             adaptive = true, dt = 0.1f-1, abstol = 1.0f-8, reltol = 1.0f-5)

## Callbacks
@info "Callbacks"

function f(u, p, t)
    du1 = -u[1]
    return SVector{1}(du1)
end

u0 = @SVector [10.0f0]
prob = ODEProblem{false}(f, u0, (0.0f0, 100.0f0))
prob_func = (prob, i, repeat) -> remake(prob, p = prob.p)
monteprob = EnsembleProblem(prob, safetycopy = false)

condition(u, t, integrator) = t == 4.0f0

affect!(integrator) = integrator.u += @SVector[10.0f0]

gpu_cb = GPUDiscreteCallback(condition, affect!)
cb = DiscreteCallback(condition, affect!)

sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(),
            trajectories = 2,
            adaptive = false, dt = 1.0f0, callback = gpu_cb, merge_callbacks = true,
            tstops = CuArray([4.0f0]))

bench_sol = solve(prob, Tsit5(),
                  adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
                  tstops = [4.0f0])

@test norm(bench_sol(4.0f0) - sol[1](4.0f0)) < 1e-6
@test norm(bench_sol.u - sol[1].u) < 3e-5

@info "Callback: CallbackSets"

condition_1(u, t, integrator) = t == 24.0f0

condition_2(u, t, integrator) = t == 40.0f0

affect!(integrator) = integrator.u += @SVector[10.0f0]

gpu_cb_1 = GPUDiscreteCallback(condition_1, affect!)
gpu_cb_2 = GPUDiscreteCallback(condition_2, affect!)

gpu_cb = CallbackSet(gpu_cb_1, gpu_cb_2)

cb_1 = DiscreteCallback(condition_1, affect!)
cb_2 = DiscreteCallback(condition_2, affect!)

cb = CallbackSet(cb_1, cb_2)

sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(),
            trajectories = 2,
            adaptive = false, dt = 1.0f0, callback = gpu_cb, merge_callbacks = true,
            tstops = [24.0f0, 40.0f0])

bench_sol = solve(prob, Tsit5(),
                  adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
                  tstops = [24.0f0, 40.0f0])

@test norm(bench_sol(24.0f0) - sol[1](24.0f0)) < 1e-6
@test norm(bench_sol(40.0f0) - sol[1](40.0f0)) < 1e-6
@test norm(bench_sol.u - sol[1].u) < 2e-5

# Float64 Tests
