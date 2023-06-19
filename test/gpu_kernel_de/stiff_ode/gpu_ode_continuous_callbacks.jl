using DiffEqGPU, StaticArrays, OrdinaryDiffEq, LinearAlgebra

include("../../utils.jl")

function f(u, p, t)
    du1 = u[2]
    du2 = -p[1]
    return SVector{2}(du1, du2)
end

function f_jac(u, p, t)
    J11 = 0.0
    J21 = 0.0
    J12 = 1.0
    J22 = 0.0
    return SMatrix{2, 2, eltype(u)}(J11, J21, J12, J22)
end

function f_tgrad(u, p, t)
    return SVector{2, eltype(u)}(0.0, 0.0)
end

u0 = @SVector[45.0f0, 0.0f0]
tspan = (0.0f0, 16.5f0)
p = @SVector [10.0f0]

func = ODEFunction(f, jac = f_jac, tgrad = f_tgrad)
prob = ODEProblem{false}(func, u0, tspan, p)

prob_func = (prob, i, repeat) -> remake(prob, p = prob.p)
monteprob = EnsembleProblem(prob, safetycopy = false)

function affect!(integrator)
    integrator.u += @SVector[0.0f0, -2.0f0] .* integrator.u
end

function condition(u, t, integrator)
    u[1]
end

algs = [GPURosenbrock23(), GPURodas4()]

for alg in algs
    @info typeof(alg)

    cb = ContinuousCallback(condition, affect!; save_positions = (false, false))

    @info "Unadaptive version"

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = false, dt = 0.1f0, callback = cb, merge_callbacks = true)

    bench_sol = solve(prob, Rosenbrock23(),
        adaptive = false, dt = 0.1f0, callback = cb, merge_callbacks = true)

    @test norm(bench_sol.u - sol[1].u) < 8e-4

    @info "Callback: CallbackSets"

    cb = CallbackSet(cb, cb)

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = false, dt = 0.1f0, callback = cb, merge_callbacks = true)

    bench_sol = solve(prob, Rosenbrock23(),
        adaptive = false, dt = 0.1f0, callback = cb, merge_callbacks = true)

    @test norm(bench_sol.u - sol[1].u) < 8e-4

    @info "saveat and callbacks"

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
        saveat = [0.0f0, 9.1f0])

    bench_sol = solve(prob, Rosenbrock23(),
        adaptive = false, dt = 1.0f0, callback = cb, merge_callbacks = true,
        saveat = [0.0f0, 9.1f0])

    @test norm(bench_sol.u - sol[1].u) < 5e-4

    @info "save_everystep and callbacks"

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = false, dt = 0.1f0, callback = cb, merge_callbacks = true,
        save_everystep = false)

    bench_sol = solve(prob, Rosenbrock23(),
        adaptive = false, dt = 0.1f0, callback = cb, merge_callbacks = true,
        save_everystep = false)

    @test norm(bench_sol.u - sol[1].u) < 6e-4

    @info "Adaptive version"

    cb = ContinuousCallback(condition, affect!; save_positions = (false, false))

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = true, dt = 0.1f0, callback = cb, merge_callbacks = true)

    bench_sol = solve(prob, Rosenbrock23(),
        adaptive = true, save_everystep = false, dt = 0.1f0, callback = cb,
        merge_callbacks = true)

    @test norm(bench_sol.u - sol[1].u) < 2e-3

    @info "Callback: CallbackSets"

    cb = CallbackSet(cb, cb)

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = true, dt = 0.1f0, callback = cb, merge_callbacks = true)

    bench_sol = solve(prob, Rosenbrock23(),
        adaptive = true, dt = 0.1f0, save_everystep = false, callback = cb,
        merge_callbacks = true)

    @test norm(bench_sol.u - sol[1].u) < 2e-3

    @info "saveat and callbacks"

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = true, dt = 0.1f0, callback = cb, merge_callbacks = true,
        saveat = [0.0f0, 9.1f0], reltol = 1.0f-6, abstol = 1.0f-6)

    bench_sol = solve(prob, Rosenbrock23(),
        adaptive = true, save_everystep = false, dt = 0.1f0, callback = cb,
        merge_callbacks = true,
        tstops = [24.0f0, 40.0f0], saveat = [0.0f0, 9.1f0], reltol = 1.0f-6,
        abstol = 1.0f-6)

    @test norm(bench_sol.u - sol[1].u) < 6e-4

    @info "Unadaptive and Adaptive comparison"

    local sol = solve(monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = false, dt = 0.1f0, callback = cb, merge_callbacks = true,
        saveat = [0.0f0, 9.1f0])

    asol = solve(monteprob, alg, EnsembleGPUKernel(backend),
        trajectories = 2,
        adaptive = true, dt = 0.1f0, callback = cb, merge_callbacks = true,
        saveat = [0.0f0, 9.1f0])

    @test norm(asol[1].u - sol[1].u) < 7e-4
end
