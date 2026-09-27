using DiffEqGPU, KernelAbstractions, SciMLBase, StaticArrays, Test

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
    # Stopping at the tstop goes through the dense output, which for Float32 Vern7
    # is ~60 ulps off even for u' = 1: https://github.com/SciML/DiffEqGPU.jl/issues/554
    if alg isa GPUVern7
        @test all(s -> all(isfinite, s.u[end]), sol.u)
    else
        @test all(s -> isapprox(s.u[end], SVector(1.25f0, 1.25f0); rtol = 1.0f-6), sol.u)
    end
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
    # u0 + (tf - t0) + 10 from the callback; Vern7 is only checked for
    # the callback having fired until issue 554 is fixed.
    if alg isa GPUVern7
        @test all(s -> all(>(11.0f0), s.u[end]), sol.u)
    else
        @test all(s -> isapprox(s.u[end], SVector(11.25f0, 11.25f0); rtol = 1.0f-6), sol.u)
    end
end

# At t ≈ 2^26 a Float32 ulp is 8: the first step (32) ends one ulp past the stop in
# time but has integrated 8 time units beyond it, so the state at the stop has to
# come from the dense output. `tf` is chosen so every later step is a whole number of
# ulps; otherwise `t + dt` rounding would decouple time from the integrated length.
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
