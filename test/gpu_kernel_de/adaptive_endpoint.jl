using DiffEqGPU, KernelAbstractions, LinearAlgebra, SciMLBase, StaticArrays, Test

const ADAPTIVE_ALGS = (
    GPUTsit5(), GPUTsit5IController(), GPUVern7(), GPUVern9(),
    GPURosenbrock23(), GPURodas4(), GPURodas5P(), GPUKvaerno3(), GPUKvaerno5(),
)

# Drives `step!` directly with an iteration cap so that an integrator that stops
# advancing before `tf` fails the test instead of hanging the kernel loop.
function reaches_endpoint(alg, ::Type{T}, tf, λ, reltol) where {T}
    f = ODEFunction{false}((u, p, t) -> SVector(-u[1], -p[1] * u[2]))
    integ = DiffEqGPU.init(
        alg, f, false, SVector{2, T}(1, 1), zero(T), T(tf), T(tf), SVector{1, T}(λ),
        T(reltol / 1000), T(reltol), DiffEqGPU.DiffEqBase.ODE_DEFAULT_NORM, nothing,
        CallbackSet(nothing), nothing
    )
    ts = zeros(T, 2)
    us = zeros(SVector{2, T}, 2)
    for _ in 1:10_000
        integ.t < T(tf) || return integ.t == T(tf)
        DiffEqGPU.step!(integ, ts, us)
    end
    return false
end

@testset "Adaptive endpoint termination ($(nameof(typeof(alg))), $T)" for
    alg in ADAPTIVE_ALGS,
        T in (Float32, Float64)
    @test all(
        reaches_endpoint(alg, T, tf, λ, reltol)
            for tf in range(0.05, 2.0; length = 200), λ in (1, 10, 100),
            reltol in (1.0e-3, 1.0e-4, 1.0e-6)
    )
end

# `t0 + (tf - t0)` rounds one ulp below `tf` here. A step that already covers the
# interval must snap to `tf` rather than take a further ulp-sized step, which would
# leave `t == tf` but shift the state of this exactly integrable problem by `eps(tf)`.
@testset "Adaptive endpoint snap ($(nameof(typeof(alg))))" for
    alg in ADAPTIVE_ALGS
    t0, tf = 2.094426f7, 9.4428696f7
    @assert t0 + (tf - t0) != tf
    f = ODEFunction{false}((u, p, t) -> SVector(1.0f0, 1.0f0))
    integ = DiffEqGPU.init(
        alg, f, false, SVector(1.0f0, 1.0f0), t0, tf, tf - t0, SVector(0.0f0),
        1.0f-9, 1.0f-6, DiffEqGPU.DiffEqBase.ODE_DEFAULT_NORM, nothing,
        CallbackSet(nothing), nothing
    )
    ts, us = zeros(Float32, 2), zeros(SVector{2, Float32}, 2)
    nsteps = 0
    while integ.t < tf && nsteps < 10
        DiffEqGPU.step!(integ, ts, us)
        nsteps += 1
    end
    @test integ.t == tf
    @test nsteps == 1
    if alg isa Union{GPUTsit5, GPUTsit5IController, GPUVern7, GPUVern9, GPURosenbrock23, GPUKvaerno5}
        @test integ.u == SVector(1.0f0, 1.0f0) .+ (tf - t0)
    end
end

# The first step covers the whole interval and `t0 + dt` rounds to `tf`, but the
# step must stop at the intermediate tstop, so the next step size has to be
# computed from the stop, not from `tf`.
@testset "Adaptive tstop before endpoint ($(nameof(typeof(alg))))" for alg in ADAPTIVE_ALGS
    t0, tf, stop = 0.027657658f0, 1.0f0, 0.5f0
    dt0 = prevfloat(tf - t0)
    @assert t0 + dt0 == tf
    f = ODEFunction{false}((u, p, t) -> SVector(1.0f0, 1.0f0))
    integ = DiffEqGPU.init(
        alg, f, false, SVector(1.0f0, 1.0f0), t0, tf, dt0, SVector(0.0f0),
        1.0f-9, 1.0f-6, DiffEqGPU.DiffEqBase.ODE_DEFAULT_NORM, Float32[stop],
        CallbackSet(nothing), nothing
    )
    ts, us = zeros(Float32, 2), zeros(SVector{2, Float32}, 2)
    visited_stop = false
    positive_dtnew = true
    nsteps = 0
    while integ.t < tf && nsteps < 100
        DiffEqGPU.step!(integ, ts, us)
        nsteps += 1
        visited_stop |= integ.t == stop
        integ.t < tf && (positive_dtnew &= integ.dtnew > 0)
    end
    @test visited_stop
    @test positive_dtnew
    @test integ.t == tf
end

@testset "Adaptive tstop before endpoint, public solve ($(nameof(typeof(alg))))" for
    alg in ADAPTIVE_ALGS
    t0, tf = 0.75f0, 1.0f0
    prob = ODEProblem{false}((u, p, t) -> SVector(1.0f0, 1.0f0), SVector(1.0f0, 1.0f0), (t0, tf))
    sol = solve(
        EnsembleProblem(prob), alg, EnsembleGPUKernel(KernelAbstractions.CPU(), 0.0);
        trajectories = 2, adaptive = true, dt = prevfloat(tf - t0),
        tstops = Float32[0.875], abstol = 1.0f-9, reltol = 1.0f-6, save_everystep = false
    )
    @test all(s -> s.t[end] == tf, sol.u)
    @test all(s -> isapprox(s.u[end], SVector(1.25f0, 1.25f0); rtol = 1.0f-6), sol.u)
end

# After landing on the first stop, the next proposed step reaches `tf`; the
# remaining stops must still be visited in order rather than snapped over.
@testset "Adaptive multiple tstops visited in order ($(nameof(typeof(alg))))" for
    alg in ADAPTIVE_ALGS
    t0, tf = 0.75f0, 1.0f0
    stops = Float32[0.8125, 0.875, 0.9375]
    f = ODEFunction{false}((u, p, t) -> SVector(1.0f0, 1.0f0))
    integ = DiffEqGPU.init(
        alg, f, false, SVector(1.0f0, 1.0f0), t0, tf, prevfloat(tf - t0), SVector(0.0f0),
        1.0f-9, 1.0f-6, DiffEqGPU.DiffEqBase.ODE_DEFAULT_NORM, stops,
        CallbackSet(nothing), nothing
    )
    ts, us = zeros(Float32, 2), zeros(SVector{2, Float32}, 2)
    times = Float32[]
    while integ.t < tf && length(times) < 100
        DiffEqGPU.step!(integ, ts, us)
        push!(times, integ.t)
    end
    @test filter(in(stops), times) == stops
    @test issorted(times)
    @test integ.t == tf
end

@testset "Adaptive callback at intermediate tstop, public solve ($(nameof(typeof(alg))))" for
    alg in ADAPTIVE_ALGS
    condition(u, t, integrator) = t == 0.875f0
    affect!(integrator) = (integrator.u += SVector(10.0f0, 10.0f0))
    cb = DiscreteCallback(condition, affect!; save_positions = (false, false))
    prob = ODEProblem{false}(
        (u, p, t) -> SVector(1.0f0, 1.0f0), SVector(1.0f0, 1.0f0), (0.75f0, 1.0f0)
    )
    sol = solve(
        EnsembleProblem(prob), alg, EnsembleGPUKernel(KernelAbstractions.CPU(), 0.0);
        trajectories = 2, adaptive = true, dt = prevfloat(0.25f0),
        tstops = Float32[0.8125, 0.875, 0.9375], callback = cb, merge_callbacks = true,
        abstol = 1.0f-9, reltol = 1.0f-6, save_everystep = false
    )
    @test all(s -> s.t[end] == 1.0f0, sol.u)
    # u0 + (tf - t0) + 10 from the callback
    @test all(s -> isapprox(s.u[end], SVector(11.25f0, 11.25f0); rtol = 1.0f-6), sol.u)
end

# At t ≈ 2^26 a Float32 ulp is 8, so the initial step (32) would end one ulp past the
# stop in time while having integrated 8 time units beyond it. `tf` keeps every step
# a whole number of ulps (see https://github.com/SciML/DiffEqGPU.jl/issues/567).
@testset "Adaptive tstop inside a coarse step, public solve ($(nameof(typeof(alg))))" for
    alg in ADAPTIVE_ALGS
    t0 = Float32(2^26)
    tf = t0 + 32.0f0
    prob = ODEProblem{false}(
        (u, p, t) -> SVector(1.0f0, 1.0f0), SVector(0.0f0, 0.0f0), (t0, tf)
    )
    sol = solve(
        EnsembleProblem(prob), alg, EnsembleGPUKernel(KernelAbstractions.CPU(), 0.0);
        trajectories = 2, adaptive = true, dt = 32.0f0, tstops = Float32[t0 + 24.0f0],
        abstol = 1.0f-6, reltol = 1.0f-3, save_everystep = false
    )
    @test all(s -> s.t[end] == tf, sol.u)
    @test all(s -> s.u[end] ≈ SVector(32.0f0, 32.0f0), sol.u)
end

# Stops closer together than the minimum step size must both be visited without
# clamping the next step below `dtmin`.
@testset "Adaptive adjacent tstops with callbacks, public solve ($(nameof(typeof(alg))))" for
    alg in ADAPTIVE_ALGS
    stops = [0.8125, nextfloat(0.8125)]
    cb = DiscreteCallback(
        (u, t, integrator) -> t in stops,
        integrator -> (integrator.u += SVector(10.0, 10.0));
        save_positions = (false, false)
    )
    prob = ODEProblem{false}((u, p, t) -> SVector(1.0, 1.0), SVector(0.0, 0.0), (0.75, 2.0))
    sol = solve(
        EnsembleProblem(prob), alg, EnsembleGPUKernel(KernelAbstractions.CPU(), 0.0);
        trajectories = 2, adaptive = true, dt = 0.0625, tstops = stops, callback = cb,
        merge_callbacks = true, abstol = 1.0e-9, reltol = 1.0e-6, save_everystep = false
    )
    @test all(s -> s.t[end] == 2.0, sol.u)
    @test all(s -> s.u[end] ≈ SVector(21.25, 21.25), sol.u)
end

# Stops at or within the minimum step size of `tf` must still be visited and their
# callbacks run; only the forced final interval to `tf` may be shorter than `dtmin`.
@testset "Adaptive tstops at the endpoint ($(nameof(typeof(alg))), $label)" for
    alg in ADAPTIVE_ALGS,
        (label, stops) in (
            ("at tf", [2.0]), ("one ulp before tf", [prevfloat(2.0)]),
            ("pair within dtmin of tf", [2.0 - 1.5e-14, 2.0 - 5.0e-15]),
            ("one ulp before and at tf", [prevfloat(2.0), 2.0]),
        )
    cb = DiscreteCallback(
        (u, t, integrator) -> t in stops,
        integrator -> (integrator.u += SVector(10.0, 10.0));
        save_positions = (false, false)
    )
    prob = ODEProblem{false}((u, p, t) -> SVector(1.0, 1.0), SVector(0.0, 0.0), (0.75, 2.0))
    sol = solve(
        EnsembleProblem(prob), alg, EnsembleGPUKernel(KernelAbstractions.CPU(), 0.0);
        trajectories = 2, adaptive = true, dt = 0.0625, tstops = stops, callback = cb,
        merge_callbacks = true, abstol = 1.0e-9, reltol = 1.0e-6, save_everystep = false
    )
    expected = 1.25 + 10 * length(stops)
    @test all(s -> s.t[end] == 2.0, sol.u)
    @test all(s -> isapprox(s.u[end], SVector(expected, expected); rtol = 1.0e-6), sol.u)
end

@testset "Adaptive terminating callback one ulp before tf ($(nameof(typeof(alg))))" for
    alg in ADAPTIVE_ALGS
    stop = prevfloat(2.0)
    cb = DiscreteCallback(
        (u, t, integrator) -> t == stop,
        integrator -> (integrator.u += SVector(10.0, 10.0); terminate!(integrator));
        save_positions = (false, false)
    )
    prob = ODEProblem{false}((u, p, t) -> SVector(1.0, 1.0), SVector(0.0, 0.0), (0.75, 2.0))
    sol = solve(
        EnsembleProblem(prob), alg, EnsembleGPUKernel(KernelAbstractions.CPU(), 0.0);
        trajectories = 2, adaptive = true, dt = 1.25 - 2.0e-14, tstops = [stop],
        callback = cb, merge_callbacks = true, abstol = 1.0e-9, reltol = 1.0e-6,
        save_everystep = false
    )
    expected = stop - 0.75 + 10
    @test all(s -> s.t[end] == stop, sol.u)
    @test all(s -> isapprox(s.u[end], SVector(expected, expected); rtol = 1.0e-6), sol.u)
end

# The step that reaches a stop must keep its dense output intact for the saveat
# points it covers: one ulp at these times is 8, so a step that overshoots the stop
# and is then overwritten with the stop's state would corrupt the sample at t0 + 8.
@testset "Adaptive saveat inside a step that reaches a tstop ($(nameof(typeof(alg))), $T)" for
    alg in ADAPTIVE_ALGS, T in (Float32, Float64)
    t0 = T(T === Float32 ? 2^26 : 2^55)
    @assert eps(t0) == T(8)
    prob = ODEProblem{false}(
        (u, p, t) -> SVector(one(T), one(T)), SVector(zero(T), zero(T)), (t0, t0 + T(32))
    )
    sol = solve(
        EnsembleProblem(prob), alg, EnsembleGPUKernel(KernelAbstractions.CPU(), 0.0);
        trajectories = 2, adaptive = true, dt = T(32), tstops = T[t0 + T(24)],
        saveat = T[t0, t0 + T(8)], abstol = T(1.0e-6), reltol = T(1.0e-3),
        save_everystep = false
    )
    @test all(s -> s.u[2] ≈ SVector(T(8), T(8)), sol.u)
end

# A final interval shorter than `dtmin` is only exempt from the `dtmin` check as a
# whole. If error control rejects it, the solve must either fail with `dt<dtmin` or
# integrate the full interval; it must never snap to `tf` after a partial step.
function final_interval_outcome(solve_thunk, exact)
    sol = try
        solve_thunk()
    catch err
        return err isa ErrorException && occursin("dt<dtmin", err.msg)
    end
    return all(s -> isapprox(s.u[end][end], exact; rtol = 1.0e-5, atol = 1.0e-9), sol.u)
end

@testset "Adaptive rejected sub-dtmin final interval ($(nameof(typeof(alg))))" for
    alg in ADAPTIVE_ALGS
    prob = ODEProblem{false}((u, p, t) -> -1.0e15 * u, SVector(1.0, 1.0), (0.0, 4.0e-15))
    @test final_interval_outcome(exp(-4.0)) do
        solve(
            EnsembleProblem(prob), alg, EnsembleGPUKernel(KernelAbstractions.CPU(), 0.0);
            trajectories = 2, adaptive = true, dt = 4.0e-15, abstol = 1.0e-9,
            reltol = 1.0e-6, save_everystep = false
        )
    end
end

@testset "Adaptive stiff dynamics switched on just before tf ($(nameof(typeof(alg))))" for
    alg in ADAPTIVE_ALGS
    stop, tf = 1.0 - 4.0e-15, 1.0
    cb = DiscreteCallback(
        (u, t, integrator) -> t == stop,
        integrator -> (integrator.u = SVector(1.0e15, 1.0));
        save_positions = (false, false)
    )
    prob = ODEProblem{false}(
        (u, p, t) -> SVector(0.0, -u[1] * u[2]), SVector(0.0, 1.0), (0.0, tf)
    )
    @test final_interval_outcome(exp(-1.0e15 * (tf - stop))) do
        solve(
            EnsembleProblem(prob), alg, EnsembleGPUKernel(KernelAbstractions.CPU(), 0.0);
            trajectories = 2, adaptive = true, dt = 0.5, tstops = [stop], callback = cb,
            merge_callbacks = true, abstol = 1.0e-9, reltol = 1.0e-6, save_everystep = false
        )
    end
end

# Property: whatever the interval length, stiffness, initial step and stops near the
# end, an adaptive solve either fails with `dt<dtmin` or reaches the analytic endpoint.
# Integrating only part of the final interval and reporting success shows up as an
# error far above the bound: across this grid every method's global error stays below
# 2000 * reltol (Rosenbrock23, second order, is the largest at about 900 * reltol).
@testset "Adaptive endpoint property grid ($(nameof(typeof(alg))))" for alg in ADAPTIVE_ALGS
    decay(u, p, t) = -p[1] * u
    failures = []
    for reltol in (1.0e-6, 1.0e-9), L in (1.0e-13, 1.0e-3, 1.0), λL in (0.1, 1.0, 4.0),
            dtfrac in (0.5, 0.999, 1.0),
            stops in ([], [L / 4], [L * (1 - 1.0e-2)], [prevfloat(L)], [L - 5.0e-15])

        stops = Float64[s for s in stops if 0 < s < L]
        prob = ODEProblem{false}(decay, SVector(1.0, 1.0), (0.0, L), SVector(λL / L))
        endpoint = try
            sol = solve(
                EnsembleProblem(prob), alg, EnsembleGPUKernel(KernelAbstractions.CPU(), 0.0);
                trajectories = 2, adaptive = true, dt = dtfrac * L, tstops = stops,
                abstol = reltol / 1000, reltol, save_everystep = false
            )
            sol.u[1].u[end][1]
        catch err
            err isa ErrorException && occursin("dt<dtmin", err.msg) && continue
            rethrow()
        end
        abs(endpoint - exp(-λL)) <= 2000 * reltol * exp(-λL) ||
            push!(failures, (; reltol, L, λL, dtfrac, stops, endpoint))
    end
    @test isempty(failures)
end

# `tstops` given as a Float64 array for a Float32 problem: the step bound must stay in
# the integrator's time type.
@testset "Adaptive Float64 tstops on a Float32 problem ($(nameof(typeof(alg))))" for
    alg in ADAPTIVE_ALGS
    prob = ODEProblem{false}(
        (u, p, t) -> SVector(1.0f0, 1.0f0), SVector(1.0f0, 1.0f0), (0.75f0, 1.0f0)
    )
    sol = solve(
        EnsembleProblem(prob), alg, EnsembleGPUKernel(KernelAbstractions.CPU(), 0.0);
        trajectories = 2, adaptive = true, dt = 0.125f0, tstops = [0.875],
        abstol = 1.0f-9, reltol = 1.0f-6, save_everystep = false
    )
    @test all(s -> s.t[end] == 1.0f0 && s.u[end] ≈ SVector(1.25f0, 1.25f0), sol.u)
end

# A stop less than `dtmin` after `t` must be reached by an integrated step, not by
# evaluating the dense output of a longer step near Θ = 1 (inaccurate for Float32
# Verner interpolants). `u' = (1e14, 0)` is integrated exactly by every method.
@testset "Adaptive tstop closer than dtmin, public solve ($(nameof(typeof(alg))))" for
    alg in ADAPTIVE_ALGS
    dt = 1.0f-14
    stop = prevfloat(dt)
    cb = DiscreteCallback(
        (u, t, integrator) -> t == stop, terminate!; save_positions = (false, false)
    )
    prob = ODEProblem{false}(
        (u, p, t) -> SVector(1.0f14, 0.0f0), SVector(0.0f0, 1.0f0), (0.0f0, 2dt)
    )
    sol = solve(
        EnsembleProblem(prob), alg, EnsembleGPUKernel(KernelAbstractions.CPU(), 0.0);
        trajectories = 2, adaptive = true, dt, tstops = [stop], callback = cb,
        merge_callbacks = true, abstol = 1.0f-9, reltol = 1.0f-6, save_everystep = false
    )
    exact = Float64(BigFloat(1.0f14) * BigFloat(stop))
    @test all(s -> s.t[end] == stop && abs(s.u[end][1] - exact) <= 64eps(Float32), sol.u)
end

# Landing on a stop far below `dtmin` keeps the state accurate for every method family,
# including Rosenbrock methods whose W = I/(γ dt) - J then has entries near 1/dt.
@testset "Adaptive tiny landing step ($(nameof(typeof(alg))), rate $rate)" for
    alg in ADAPTIVE_ALGS, rate in (1.0f14, 1.0f20)
    stop = 1.0f-20
    cb = DiscreteCallback(
        (u, t, integrator) -> t == stop, terminate!; save_positions = (false, false)
    )
    prob = ODEProblem{false}(
        (u, p, t) -> SVector(rate, rate), SVector(0.0f0, 0.0f0), (0.0f0, 2.0f-14)
    )
    sol = solve(
        EnsembleProblem(prob), alg, EnsembleGPUKernel(KernelAbstractions.CPU(), 0.0);
        trajectories = 2, adaptive = true, dt = 1.0f-14, tstops = [stop], callback = cb,
        merge_callbacks = true, save_everystep = false, abstol = 1.0f-12, reltol = 1.0f-3
    )
    exact = Float64(BigFloat(rate) * BigFloat(stop))
    @test all(
        s -> s.t[end] == stop &&
            all(x -> isapprox(Float64(x), exact; rtol = 64eps(Float32)), s.u[end]),
        sol.u
    )
end

# A stop so close that the landing step overflows (Rosenbrock `C/dt` terms) is reached by
# interpolating the controller's covering step: the result is finite and accurate, never
# NaN. Vern7's dense output is checked only for finiteness:
# https://github.com/SciML/DiffEqGPU.jl/issues/554
@testset "Adaptive stop near floatmin ($(nameof(typeof(alg))), $T)" for
    alg in ADAPTIVE_ALGS, T in (Float32, Float64)
    stop = T === Float32 ? 5.0f-38 : 1.0e-307
    rate = T(1.0e14)
    cb = DiscreteCallback(
        (u, t, integrator) -> t == stop, terminate!; save_positions = (false, false)
    )
    prob = ODEProblem{false}(
        (u, p, t) -> SVector(rate, rate), SVector(zero(T), zero(T)), (zero(T), T(2.0e-14))
    )
    sol = solve(
        EnsembleProblem(prob), alg, EnsembleGPUKernel(KernelAbstractions.CPU(), 0.0);
        trajectories = 2, adaptive = true, dt = T(1.0e-14), tstops = [stop], callback = cb,
        merge_callbacks = true, save_everystep = false, abstol = T(1.0e-12), reltol = T(1.0e-3)
    )
    exact = Float64(BigFloat(rate) * BigFloat(stop))
    @test all(s -> s.t[end] == stop && all(isfinite, s.u[end]), sol.u)
    if !(alg isa GPUVern7)
        @test all(
            s -> all(x -> isapprox(Float64(x), exact; rtol = 128eps(T)), s.u[end]), sol.u
        )
    end
end

# With a scaled mass matrix the landing step's W = J - M/(γ dt) overflows even when the
# stop itself is representable. A step whose state or error estimate is not finite is
# never accepted: the solver retries with the controller's covering step and lands on
# the stop by interpolation.
@testset "Adaptive overflowing landing step with mass matrix ($(nameof(typeof(alg))), $T)" for
    alg in (GPURosenbrock23(), GPURodas4(), GPURodas5P()), T in (Float32, Float64)
    stop, rate, mass = T(1024) / floatmax(T), T(1.0e14), T(1024)
    f = ODEFunction{false}(
        (u, p, t) -> SVector(rate, rate); mass_matrix = mass * SMatrix{2, 2, T}(I)
    )
    prob = ODEProblem{false}(f, zero(SVector{2, T}), (zero(T), T(2.0e-14)))
    cb = DiscreteCallback(
        (u, t, integrator) -> t == stop, terminate!; save_positions = (false, false)
    )
    sol = solve(
        EnsembleProblem(prob), alg, EnsembleGPUKernel(KernelAbstractions.CPU(), 0.0);
        trajectories = 2, adaptive = true, dt = T(1.0e-14), tstops = [stop], callback = cb,
        merge_callbacks = true, save_everystep = false, abstol = T(1.0e-12), reltol = T(1.0e-3)
    )
    exact = BigFloat(rate) * BigFloat(stop) / BigFloat(mass)
    @test all(
        s -> s.t[end] == stop &&
            all(x -> isapprox(BigFloat(x), exact; rtol = 128eps(T)), s.u[end]),
        sol.u
    )
end

# `saveat` points at the ends of an accepted step are the step's own states; the dense
# output is only used strictly inside the step.
@testset "Adaptive saveat at step endpoints ($(nameof(typeof(alg))), $T)" for
    alg in ADAPTIVE_ALGS, T in (Float32, Float64)
    stop = T(0.125)
    cb = DiscreteCallback(
        (u, t, integrator) -> t == stop, terminate!; save_positions = (false, false)
    )
    prob = ODEProblem{false}(
        (u, p, t) -> SVector(one(T), one(T)), zero(SVector{2, T}), (zero(T), one(T))
    )
    sol = solve(
        EnsembleProblem(prob), alg, EnsembleGPUKernel(KernelAbstractions.CPU(), 0.0);
        trajectories = 2, adaptive = true, dt = T(0.5), tstops = [stop], callback = cb,
        merge_callbacks = true, saveat = T[0, stop / 2, stop], abstol = T(1.0e-12),
        reltol = T(1.0e-6)
    )
    @test all(s -> s.t[end] == stop && s.u[1] == zero(SVector{2, T}), sol.u)
    @test all(s -> all(x -> isapprox(x, stop; rtol = 128eps(T)), s.u[end]), sol.u)
end
