
function Adapt.adapt_structure(to, prob::ODEProblem{<:Any, <:Any, iip}) where {iip}
    ODEProblem{iip, true}(adapt(to, prob.f),
                          adapt(to, prob.u0),
                          adapt(to, prob.tspan),
                          adapt(to, prob.p);
                          adapt(to, prob.kwargs)...)
end

## Fixed TimeStep Integrator
mutable struct GPUTsit5Integrator{IIP, S, T, P, F, TS, CB} <:
               DiffEqBase.AbstractODEIntegrator{GPUTsit5, IIP, S, T}
    f::F                  # eom
    uprev::S              # previous state
    u::S                  # current state
    tmp::S                # dummy, same as state
    tprev::T              # previous time
    t::T                  # current time
    t0::T                 # initial time, only for reinit
    dt::T                 # step size
    tdir::T
    p::P                  # parameter container
    u_modified::Bool
    tstops::TS
    tstops_idx::Int
    cb::CB
    save_everystep::Bool
    step_idx::Int
    k1::S                 #intepolants
    k2::S
    k3::S
    k4::S
    k5::S
    k6::S
    k7::S
    cs::SVector{6, T}     # ci factors cache: time coefficients
    as::SVector{21, T}    # aij factors cache: solution coefficients
    rs::SVector{22, T}    # rij factors cache: interpolation coefficients
end
const GPUT5I = GPUTsit5Integrator

(integrator::GPUTsit5Integrator)(t) = copy(integrator.u)
(integrator::GPUTsit5Integrator)(out, t) = (out .= integrator.u)

function DiffEqBase.u_modified!(integrator::GPUTsit5Integrator, bool::Bool)
    integrator.u_modified = bool
end

DiffEqBase.isinplace(::GPUT5I{IIP}) where {IIP} = IIP

## Adaptive TimeStep Integrator

mutable struct GPUATsit5Integrator{IIP, S, T, P, F, N, TOL, Q} <:
               DiffEqBase.AbstractODEIntegrator{GPUTsit5, IIP, S, T}
    f::F                  # eom
    uprev::S              # previous state
    u::S                  # current state
    tmp::S                # dummy, same as state
    tprev::T              # previous time
    t::T                  # current time
    t0::T                 # initial time, only for reinit
    tf::T
    dt::T                 # step size
    dtnew::T
    tdir::T
    p::P                  # parameter container
    u_modified::Bool
    k1::S         # interpolants of the algorithm
    k2::S
    k3::S
    k4::S
    k5::S
    k6::S
    k7::S
    cs::SVector{6, T}     # ci factors cache: time coefficients
    as::SVector{21, T}    # aij factors cache: solution coefficients
    btildes::SVector{7, T}
    rs::SVector{22, T}    # rij factors cache: interpolation coefficients
    qold::Q
    abstol::TOL
    reltol::TOL
    internalnorm::N       # function that computes the error EEst based on state
end

const GPUAT5I = GPUATsit5Integrator

#######################################################################################
# Initialization of Integrators
#######################################################################################
@inline function gputsit5_init(f::F, IIP::Bool, u0::S, t0::T, dt::T,
                               p::P, tstops::TS,
                               callback::CB,
                               save_everystep::Bool) where {F, P, T, S <: AbstractArray{T},
                                                            TS, CB}
    cs, as, rs = SimpleDiffEq._build_tsit5_caches(T)

    !IIP && @assert S <: SArray

    integ = GPUT5I{IIP, S, T, P, F, TS, CB}(f, copy(u0), copy(u0), copy(u0), t0, t0, t0, dt,
                                            sign(dt), p, true, tstops, 1, callback,
                                            save_everystep, 1,
                                            copy(u0), copy(u0), copy(u0), copy(u0),
                                            copy(u0),
                                            copy(u0), copy(u0), cs, as, rs)
end

@inline function gpuatsit5_init(f::F, IIP::Bool, u0::S, t0::T, tf::T, dt::T, p::P,
                                abstol::TOL, reltol::TOL,
                                internalnorm::N) where {F, P, S, T, N, TOL}
    cs, as, btildes, rs = SimpleDiffEq._build_atsit5_caches(T)

    !IIP && @assert S <: SArray

    qoldinit = eltype(S)(1e-4)

    integ = GPUAT5I{IIP, S, T, P, F, N, TOL, typeof(qoldinit)}(f, copy(u0), copy(u0),
                                                               copy(u0), t0, t0, t0, tf, dt,
                                                               dt, sign(tf - t0), p, true,
                                                               copy(u0), copy(u0), copy(u0),
                                                               copy(u0), copy(u0), copy(u0),
                                                               copy(u0), cs, as, btildes,
                                                               rs, qoldinit, abstol, reltol,
                                                               internalnorm)
end
## GPU solver

function vectorized_solve(probs, prob::ODEProblem, alg::GPUSimpleTsit5;
                          dt, saveat = nothing,
                          save_everystep = true,
                          debug = false, callback = nothing, tstops = nothing, kwargs...)
    # if saveat is specified, we'll use a vector of timestamps.
    # otherwise it's a matrix that may be different for each ODE.
    if saveat === nothing
        if save_everystep
            len_tstops = tstops === nothing ? 0 : length(tstops)
            len = length(prob.tspan[1]:dt:prob.tspan[2]) + len_tstops
        else
            len = 2
        end
        ts = CuMatrix{typeof(dt)}(undef, (len, length(probs)))
        us = CuMatrix{typeof(prob.u0)}(undef, (len, length(probs)))
    else
        error("Not fully implemented yet")  # see the TODO in the kernel
        ts = saveat
        us = CuMatrix{typeof(prob.u0)}(undef, (length(ts), length(ps)))
    end

    # Handle tstops
    tstops = cu(tstops)

    if callback !== nothing && !(typeof(callback) <: Tuple{})
        callback = CallbackSet(callback)
    end

    kernel = @cuda launch=false tsit5_kernel(probs, us, ts, dt, callback, tstops,
                                             Val(saveat !== nothing), Val(save_everystep))
    if debug
        @show CUDA.registers(kernel)
        @show CUDA.memory(kernel)
    end

    config = launch_configuration(kernel.fun)
    threads = min(length(probs), config.threads)
    # XXX: this kernel performs much better with all blocks active
    blocks = max(cld(length(probs), threads), config.blocks)
    threads = cld(length(probs), blocks)
    kernel(probs, us, ts, dt, callback, tstops; threads, blocks)

    # we build the actual solution object on the CPU because the GPU would create one
    # containig CuDeviceArrays, which we cannot use on the host (not GC tracked,
    # no useful operations, etc). That's unfortunate though, since this loop is
    # generally slower than the entire GPU execution, and necessitates synchronization
    #EDIT: Done when using with DiffEqGPU
    ts, us
end

function savevalues!(integrator::GPUTsit5Integrator, ts, us, force = false)
    saved, savedexactly = false, false

    if integrator.save_everystep || force
        saved = true
        savedexactly = true
        @inbounds us[integrator.step_idx] = integrator.u
        @inbounds ts[integrator.step_idx] = integrator.t
        integrator.step_idx += 1
    end

    saved, savedexactly
end

@inline function apply_discrete_callback!(integrator, ts, us,
                                          callback::GPUDiscreteCallback)
    saved_in_cb = false
    if callback.condition(integrator.u, integrator.t, integrator)
        # handle saveat
        _, savedexactly = savevalues!(integrator, ts, us)
        saved_in_cb = true
        @inbounds if callback.save_positions[1]
            # if already saved then skip saving
            savedexactly || savevalues!(integrator, ts, us, true)
        end
        integrator.u_modified = true
        callback.affect!(integrator)
        @inbounds if callback.save_positions[2]
            savevalues!(integrator, ts, us, true)
            saved_in_cb = true
        end
    end
    integrator.u_modified, saved_in_cb
end

@inline function apply_discrete_callback!(integrator, ts, us, callback::GPUDiscreteCallback,
                                          args...)
    apply_discrete_callback!(integrator, ts, us,
                             apply_discrete_callback!(integrator, ts, us, callback)...,
                             args...)
end

@inline function apply_discrete_callback!(integrator, ts, us, discrete_modified::Bool,
                                          saved_in_cb::Bool, callback::GPUDiscreteCallback,
                                          args...)
    bool, saved_in_cb2 = apply_discrete_callback!(integrator, ts, us,
                                                  apply_discrete_callback!(integrator, ts,
                                                                           us, callback)...,
                                                  args...)
    discrete_modified || bool, saved_in_cb || saved_in_cb2
end

@inline function apply_discrete_callback!(integrator, ts, us, discrete_modified::Bool,
                                          saved_in_cb::Bool, callback::GPUDiscreteCallback)
    bool, saved_in_cb2 = apply_discrete_callback!(integrator, ts, us, callback)
    discrete_modified || bool, saved_in_cb || saved_in_cb2
end

@inline function step!(integ::GPUT5I{false, S, T}, ts, us) where {T, S}
    c1, c2, c3, c4, c5, c6 = integ.cs
    dt = integ.dt
    t = integ.t
    p = integ.p
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54,
    a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76 = integ.as

    tmp = integ.tmp
    f = integ.f
    integ.uprev = integ.u
    uprev = integ.u

    integ.tprev = t
    saved_in_cb = false
    adv_integ = true
    ## Check if tstops are within the range of time-series
    if integ.tstops !== nothing && integ.tstops_idx <= length(integ.tstops) &&
       integ.tstops[integ.tstops_idx] <= integ.t + integ.dt
        integ.t = integ.tstops[integ.tstops_idx]
        integ.tstops_idx += 1
    else
        ##Advance the integrator
        integ.t += dt
    end

    if integ.u_modified
        k1 = f(uprev, p, t)
        integ.u_modified = false
    else
        @inbounds k1 = integ.k7
    end

    tmp = uprev + dt * a21 * k1
    k2 = f(tmp, p, t + c1 * dt)
    tmp = uprev + dt * (a31 * k1 + a32 * k2)
    k3 = f(tmp, p, t + c2 * dt)
    tmp = uprev + dt * (a41 * k1 + a42 * k2 + a43 * k3)
    k4 = f(tmp, p, t + c3 * dt)
    tmp = uprev + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
    k5 = f(tmp, p, t + c4 * dt)
    tmp = uprev + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
    k6 = f(tmp, p, t + dt)

    integ.u = uprev +
              dt * ((a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4) + a75 * k5 + a76 * k6)
    k7 = f(integ.u, p, t + dt)

    @inbounds begin # Necessary for interpolation
        integ.k1 = k7
        integ.k2 = k2
        integ.k3 = k3
        integ.k4 = k4
        integ.k5 = k5
        integ.k6 = k6
        integ.k7 = k7
    end

    if integ.cb !== nothing
        _, saved_in_cb = apply_discrete_callback!(integ, ts, us,
                                                  integ.cb.discrete_callbacks...)
    else
        saved_in_cb = false
    end

    return saved_in_cb
end

# saveat is just a bool here:
#  true: ts is a vector of timestamps to read from
#  false: each ODE has its own timestamps, so ts is a vector to write to
function tsit5_kernel(probs, _us, _ts, dt, callback, tstops,
                      ::Val{saveat}, ::Val{save_everystep}) where {saveat, save_everystep}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(probs) && return

    # get the actual problem for this thread
    prob = @inbounds probs[i]

    # get the input/output arrays for this thread
    ts = if saveat
        _ts
    else
        @inbounds view(_ts, :, i)
    end
    us = @inbounds view(_us, :, i)

    integ = gputsit5_init(prob.f, false, prob.u0, prob.tspan[1], dt, prob.p, tstops,
                          callback, save_everystep)

    @inbounds ts[integ.step_idx] = prob.tspan[1]
    @inbounds us[integ.step_idx] = prob.u0
    integ.step_idx += 1

    # FSAL
    while integ.step_idx <= length(ts)
        saved_in_cb = step!(integ, ts, us)
        if saveat
            # TODO
        elseif save_everystep & !saved_in_cb
            @inbounds us[integ.step_idx] = integ.u
            @inbounds ts[integ.step_idx] = integ.t
        end
        if !saved_in_cb
            integ.step_idx += 1
        end
    end

    if !saveat && !save_everystep
        @inbounds us[2] = integ.u
        @inbounds ts[2] = integ.t
    end

    return nothing
end

#############################Adaptive Version#####################################

function vectorized_asolve(probs, prob::ODEProblem, alg::GPUSimpleATsit5;
                           dt = 0.1f0, saveat = nothing,
                           save_everystep = false,
                           abstol = 1.0f-6, reltol = 1.0f-3,
                           debug = false, kwargs...)
    # if saveat is specified, we'll use a vector of timestamps.
    # otherwise it's a matrix that may be different for each ODE.
    if saveat === nothing
        if save_everystep
            error("Don't use adaptive version with saveat == nothing and save_everystep = true")
        else
            len = 2
        end
        ts = CuMatrix{typeof(dt)}(undef, (len, length(probs)))
        us = CuMatrix{typeof(prob.u0)}(undef, (len, length(probs)))
    else
        ts = saveat
        us = CuMatrix{typeof(prob.u0)}(undef, (length(ts), length(probs)))
    end

    kernel = @cuda launch=false atsit5_kernel(probs, us, ts, dt, abstol, reltol,
                                              Val(saveat !== nothing), Val(save_everystep))
    if debug
        @show CUDA.registers(kernel)
        @show CUDA.memory(kernel)
    end

    config = launch_configuration(kernel.fun)
    threads = min(length(probs), config.threads)
    # XXX: this kernel performs much better with all blocks active
    blocks = max(cld(length(probs), threads), config.blocks)
    threads = cld(length(probs), blocks)
    kernel(probs, us, ts, dt, abstol, reltol; threads, blocks)

    # we build the actual solution object on the CPU because the GPU would create one
    # containig CuDeviceArrays, which we cannot use on the host (not GC tracked,
    # no useful operations, etc). That's unfortunate though, since this loop is
    # generally slower than the entire GPU execution, and necessitates synchronization
    #EDIT: Done when using with DiffEqGPU
    ts, us
end

function build_adaptive_tsit5_controller_cache(::Type{T}) where {T}
    beta1 = T(7 / 50)
    beta2 = T(2 / 25)
    qmax = T(10.0)
    qmin = T(1 / 5)
    gamma = T(9 / 10)
    qoldinit = T(1e-4)
    qold = qoldinit

    return beta1, beta2, qmax, qmin, gamma, qoldinit, qold
end

@inline function step!(integ::GPUAT5I{false, S, T}) where {S, T}
    beta1, beta2, qmax, qmin, gamma, qoldinit, _ = build_adaptive_tsit5_controller_cache(eltype(integ.u))
    c1, c2, c3, c4, c5, c6 = integ.cs
    dt = integ.dtnew
    t = integ.t
    p = integ.p
    tf = integ.tf
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54,
    a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76 = integ.as
    btilde1, btilde2, btilde3, btilde4, btilde5, btilde6, btilde7 = integ.btildes

    tmp = integ.tmp
    f = integ.f
    integ.uprev = integ.u
    uprev = integ.u

    qold = integ.qold
    abstol = integ.abstol
    reltol = integ.reltol

    if integ.u_modified
        k1 = f(uprev, p, t)
        integ.u_modified = false
    else
        @inbounds k1 = integ.k7
    end

    EEst = Inf

    while EEst > 1.0
        dt < 1e-14 && error("dt<dtmin")

        tmp = uprev + dt * a21 * k1
        k2 = f(tmp, p, t + c1 * dt)
        tmp = uprev + dt * (a31 * k1 + a32 * k2)
        k3 = f(tmp, p, t + c2 * dt)
        tmp = uprev + dt * (a41 * k1 + a42 * k2 + a43 * k3)
        k4 = f(tmp, p, t + c3 * dt)
        tmp = uprev + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
        k5 = f(tmp, p, t + c4 * dt)
        tmp = uprev + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
        k6 = f(tmp, p, t + dt)
        u = uprev + dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6)
        k7 = f(u, p, t + dt)

        tmp = dt * (btilde1 * k1 + btilde2 * k2 + btilde3 * k3 + btilde4 * k4 +
               btilde5 * k5 + btilde6 * k6 + btilde7 * k7)
        tmp = tmp ./ (abstol .+ max.(abs.(uprev), abs.(u)) * reltol)
        EEst = DiffEqBase.ODE_DEFAULT_NORM(tmp, t)

        if iszero(EEst)
            q = inv(qmax)
        else
            q11 = EEst^beta1
            q = q11 / (qold^beta2)
        end

        if EEst > 1
            dt = dt / min(inv(qmin), q11 / gamma)
        else # EEst <= 1
            q = max(inv(qmax), min(inv(qmin), q / gamma))
            qold = max(EEst, qoldinit)
            dtnew = dt / q #dtnew
            dtnew = min(abs(dtnew), abs(tf - t - dt))

            @inbounds begin # Necessary for interpolation
                integ.k1 = k1
                integ.k2 = k2
                integ.k3 = k3
                integ.k4 = k4
                integ.k5 = k5
                integ.k6 = k6
                integ.k7 = k7
            end

            integ.dt = dt
            integ.dtnew = dtnew
            integ.qold = qold
            integ.tprev = t
            integ.u = u

            if (tf - t - dt) < 1e-14
                integ.t = tf
            else
                integ.t += dt
            end
        end
    end
    return nothing
end

function atsit5_kernel(probs, _us, _ts, dt, abstol, reltol,
                       ::Val{saveat}, ::Val{save_everystep}) where {saveat, save_everystep}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(probs) && return

    # get the actual problem for this thread
    prob = @inbounds probs[i]
    # get the input/output arrays for this thread
    ts = if saveat
        _ts
    else
        @inbounds view(_ts, :, i)
    end
    us = @inbounds view(_us, :, i)
    # TODO: optimize contiguous view to return a CuDeviceArray

    u0 = prob.u0
    tspan = prob.tspan
    f = prob.f
    p = prob.p

    t = tspan[1]
    tf = prob.tspan[2]

    cur_t = 0
    if saveat !== nothing
        cur_t = 1
        if tspan[1] == ts[1]
            cur_t += 1
            @inbounds us[1] = u0
        end
    else
        @inbounds ts[1] = tspan[1]
        @inbounds us[1] = u0
    end

    integ = gpuatsit5_init(prob.f, false, prob.u0, prob.tspan[1], prob.tspan[2], dt, prob.p,
                           abstol, reltol, DiffEqBase.ODE_DEFAULT_NORM)

    while integ.t < tspan[2]
        step!(integ)
        if saveat === nothing && save_everystep
            error("Do not use saveat == nothing & save_everystep = true in adaptive version")
        else
            saveat !== nothing
            while cur_t <= length(ts) && ts[cur_t] <= integ.t
                savet = ts[cur_t]
                θ = (savet - integ.tprev) / integ.dt
                b1θ, b2θ, b3θ, b4θ, b5θ, b6θ, b7θ = SimpleDiffEq.bθs(integ.rs, θ)
                us[cur_t] = integ.uprev +
                            integ.dt * (b1θ * integ.k1 + b2θ * integ.k2 + b3θ * integ.k3 +
                             b4θ * integ.k4 + b5θ * integ.k5 + b6θ * integ.k6 +
                             b7θ * integ.k7)
                cur_t += 1
            end
        end
    end

    if !saveat && !save_everystep
        @inbounds us[2] = integ.u
        @inbounds ts[2] = integ.t
    end

    return nothing
end
