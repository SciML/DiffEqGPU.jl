using DiffEqGPU, OrdinaryDiffEq, StaticArrays, LinearAlgebra, CUDA
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

cb = DiscreteCallback(condition, affect!; save_positions = (false, false))

@info "Unadaptive version"

sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(),
            trajectories = 2,
            adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
            tstops = [4.0f0])

bench_sol = solve(prob, Tsit5(),
                  adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
                  tstops = [4.0f0])

@test norm(bench_sol(4.0f0) - sol[1](4.0f0)) < 1e-6
@test norm(bench_sol.u - sol[1].u) < 3e-5

#Test the truncation error due to floating point math, encountered when adjusting t for tstops
sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(),
            trajectories = 2,
            adaptive = false, dt = 0.01f0, callback = cb, merge_callbacks = true,
            tstops = [4.0f0])

bench_sol = solve(prob, Tsit5(),
                  adaptive = false, dt = 0.01f0, callback = cb, merge_callbacks = true,
                  tstops = [4.0f0])

@test norm(bench_sol(4.0f0) - sol[1](4.0f0)) < 1e-6
@test norm(bench_sol.u - sol[1].u) < 3e-5

@info "Callback: CallbackSets"

condition_1(u, t, integrator) = t == 24.0f0

condition_2(u, t, integrator) = t == 40.0f0

cb_1 = DiscreteCallback(condition_1, affect!; save_positions = (false, false))
cb_2 = DiscreteCallback(condition_2, affect!; save_positions = (false, false))

cb = CallbackSet(cb_1, cb_2)

sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(),
            trajectories = 2,
            adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
            tstops = [24.0f0, 40.0f0])

bench_sol = solve(prob, Tsit5(),
                  adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
                  tstops = [24.0f0, 40.0f0])

@test norm(bench_sol(24.0f0) - sol[1](24.0f0)) < 1e-6
@test norm(bench_sol(40.0f0) - sol[1](40.0f0)) < 1e-6
@test norm(bench_sol.u - sol[1].u) < 2e-5

@info "saveat and callbacks"

sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(),
            trajectories = 2,
            adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
            tstops = [24.0f0, 40.0f0], saveat = [0.0f0, 40.0f0])

bench_sol = solve(prob, Tsit5(),
                  adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
                  tstops = [24.0f0, 40.0f0], saveat = [0.0f0, 40.0f0])

@test norm(bench_sol(24.0f0) - sol[1](24.0f0)) < 1e-6
@test norm(bench_sol(40.0f0) - sol[1](40.0f0)) < 1e-6
@test norm(bench_sol.u - sol[1].u) < 2e-5

@info "save_everystep and callbacks"

sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(),
            trajectories = 2,
            adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
            tstops = [24.0f0, 40.0f0], save_everystep = false)

bench_sol = solve(prob, Tsit5(),
                  adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
                  tstops = [24.0f0, 40.0f0], save_everystep = false)

@test norm(bench_sol(24.0f0) - sol[1](24.0f0)) < 1e-6
@test norm(bench_sol(40.0f0) - sol[1](40.0f0)) < 1e-6
@test norm(bench_sol.u - sol[1].u) < 2e-5

@info "Adaptive version"

cb = DiscreteCallback(condition, affect!; save_positions = (false, false))

sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(),
            trajectories = 2,
            adaptive = true, dt = 1.0f0, callback = cb, merge_callbacks = true,
            tstops = [4.0f0])

bench_sol = solve(prob, Tsit5(),
                  adaptive = true, save_everystep = false, dt = 1.0f0, callback = cb,
                  merge_callbacks = true,
                  tstops = [4.0f0])

@test norm(bench_sol(4.0f0) - sol[1](4.0f0)) < 8e-6
@test norm(bench_sol.u - sol[1].u) < 2e-4

@info "Callback: CallbackSets"

sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(),
            trajectories = 2,
            adaptive = true, dt = 1.0f0, callback = cb, merge_callbacks = true,
            tstops = [24.0f0, 40.0f0])

bench_sol = solve(prob, Tsit5(),
                  adaptive = true, dt = 1.0f0, save_everystep = false, callback = cb,
                  merge_callbacks = true,
                  tstops = [24.0f0, 40.0f0])

@test norm(bench_sol(24.0f0) - sol[1](24.0f0)) < 1e-6
@test norm(bench_sol(40.0f0) - sol[1](40.0f0)) < 2e-7
@test norm(bench_sol.u - sol[1].u) < 2e-5

@info "saveat and callbacks"

sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(),
            trajectories = 2,
            adaptive = true, dt = 1.0f0, callback = cb, merge_callbacks = true,
            tstops = [24.0f0, 40.0f0], saveat = [0.0f0, 40.0f0])

bench_sol = solve(prob, Tsit5(),
                  adaptive = true, save_everystep = false, dt = 1.0f0, callback = cb,
                  merge_callbacks = true,
                  tstops = [24.0f0, 40.0f0], saveat = [0.0f0, 40.0f0])

@test norm(bench_sol(24.0f0) - sol[1](24.0f0)) < 1e-6
@test norm(bench_sol(40.0f0) - sol[1](40.0f0)) < 1e-6
@test norm(bench_sol.u - sol[1].u) < 2e-5

@info "Unadaptive and Adaptive comparison"

sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(),
            trajectories = 2,
            adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
            tstops = [24.0f0, 40.0f0], saveat = [0.0f0, 40.0f0])

asol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(),
             trajectories = 2,
             adaptive = true, dt = 1.0f0, callback = cb, merge_callbacks = true,
             tstops = [24.0f0, 40.0f0], saveat = [0.0f0, 40.0f0])

@test norm(asol[1](24.0f0) - sol[1](24.0f0)) < 1e-6
@test norm(asol[1](40.0f0) - sol[1](40.0f0)) < 1e-6
@test norm(asol[1].u - sol[1].u) < 2e-5
