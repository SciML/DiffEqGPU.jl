
Adapt.adapt_structure(to, prob::ODEProblem{<:Any, <:Any, iip}) where {iip} =
    ODEProblem{iip,true}(
        adapt(to, prob.f),
        adapt(to, prob.u0),
        adapt(to, prob.tspan),
        adapt(to, prob.p);
        adapt(to, prob.kwargs)...
    )



mutable struct GPUTsit5Integrator{IIP, S, T, P, F} <: DiffEqBase.AbstractODEIntegrator{GPUTsit5, IIP, S, T}
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

DiffEqBase.isinplace(::GPUT5I{IIP}) where {IIP} = IIP

#######################################################################################
# Initialization
#######################################################################################
@inline function gputsit5_init(f::F, IIP::Bool, u0::S, t0::T, dt::T, p::P) where {F, P, T, S<:AbstractArray{T}}

    cs, as, rs = SimpleDiffEq._build_tsit5_caches(T)

    !IIP && @assert S <: SArray

    integ = GPUT5I{IIP, S, T, P, F}(f, copy(u0), copy(u0), copy(u0), t0, t0, t0, dt, sign(dt), p, true,
                                    copy(u0), copy(u0), copy(u0), copy(u0), copy(u0), copy(u0), copy(u0), cs, as, rs)
end
## GPU solver

function vectorized_solve(prob::ODEProblem, ps::CuVector, alg::GPUSimpleTsit5;
                          dt, saveat = nothing,
                          save_everystep = true,
                          debug = false, kwargs...)
    # if saveat is specified, we'll use a vector of timestamps.
    # otherwise it's a matrix that may be different for each ODE.
    if saveat === nothing
        if save_everystep
            len = length(prob.tspan[1]:dt:prob.tspan[2])
        else
            len = 2
        end
        ts = CuMatrix{typeof(dt)}(undef, (len, length(ps)))
        us = CuMatrix{typeof(prob.u0)}(undef, (len, length(ps)))
    else
        error("Not fully implemented yet")  # see the TODO in the kernel
        ts = saveat
        us = CuMatrix{typeof(prob.u0)}(undef, (length(ts), length(ps)))
    end

    
    kernel = @cuda launch=false tsit5_kernel(prob, ps, us, ts, dt,
                                             Val(saveat !== nothing), Val(save_everystep))
    if debug
        @show CUDA.registers(kernel)
        @show CUDA.memory(kernel)
    end

    config = launch_configuration(kernel.fun)
    threads = min(length(ps), config.threads)
    # XXX: this kernel performs much better with all blocks active
    blocks = max(cld(length(ps), threads), config.blocks)
    threads = cld(length(ps), blocks)
    kernel(prob, ps, us, ts, dt; threads, blocks)

    # we build the actual solution object on the CPU because the GPU would create one
    # containig CuDeviceArrays, which we cannot use on the host (not GC tracked,
    # no useful operations, etc). That's unfortunate though, since this loop is
    # generally slower than the entire GPU execution, and necessitates synchronization
    #EDIT: Done when using with DiffEqGPU
    ts,us
end

function vectorized_asolve(prob::ODEProblem, ps::CuVector, alg::GPUSimpleATsit5;
    dt=0.1f0, saveat=nothing,
    save_everystep=false,
    abstol=1.0f-6, reltol=1.0f-3,
    debug=false, kwargs...)
    # if saveat is specified, we'll use a vector of timestamps.
    # otherwise it's a matrix that may be different for each ODE.
    if saveat === nothing
        if save_everystep
            error("Don't use adaptive version with saveat == nothing and save_everystep = true")
        else
            len = 2
        end
        ts = CuMatrix{typeof(dt)}(undef, (len, length(ps)))
        us = CuMatrix{typeof(prob.u0)}(undef, (len, length(ps)))
    else
        ts = saveat
        us = CuMatrix{typeof(prob.u0)}(undef, (length(ts), length(ps)))
    end

    kernel = @cuda launch = false atsit5_kernel(prob, ps, us, ts, dt, abstol, reltol,
        Val(saveat !== nothing), Val(save_everystep))
    if debug
        @show CUDA.registers(kernel)
        @show CUDA.memory(kernel)
    end

    config = launch_configuration(kernel.fun)
    threads = min(length(ps), config.threads)
    # XXX: this kernel performs much better with all blocks active
    blocks = max(cld(length(ps), threads), config.blocks)
    threads = cld(length(ps), blocks)
    kernel(prob, ps, us, ts, dt, abstol, reltol; threads, blocks)

    # we build the actual solution object on the CPU because the GPU would create one
    # containig CuDeviceArrays, which we cannot use on the host (not GC tracked,
    # no useful operations, etc). That's unfortunate though, since this loop is
    # generally slower than the entire GPU execution, and necessitates synchronization
    #EDIT: Done when using with DiffEqGPU
    ts, us
end

@inline function step!(integ::GPUT5I{false, S, T}) where {T, S}

    c1, c2, c3, c4, c5, c6 = integ.cs;
    dt = integ.dt; t = integ.t; p = integ.p
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54,
    a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76 = integ.as

    tmp = integ.tmp; f = integ.f
    integ.uprev = integ.u; uprev = integ.u

    if integ.u_modified
      k1 = f(uprev, p, t)
      integ.u_modified=false
    else
      @inbounds k1 = integ.k7;
    end

    tmp = uprev+dt*a21*k1
    k2 = f(tmp, p, t+c1*dt)
    tmp = uprev+dt*(a31*k1+a32*k2)
    k3 = f(tmp, p, t+c2*dt)
    tmp = uprev+dt*(a41*k1+a42*k2+a43*k3)
    k4 = f(tmp, p, t+c3*dt)
    tmp = uprev+dt*(a51*k1+a52*k2+a53*k3+a54*k4)
    k5 = f(tmp, p, t+c4*dt)
    tmp = uprev+dt*(a61*k1+a62*k2+a63*k3+a64*k4+a65*k5)
    k6 = f(tmp, p, t+dt)

    integ.u = uprev+dt*((a71*k1+a72*k2+a73*k3+a74*k4)+a75*k5+a76*k6)
    k7 = f(integ.u, p, t+dt)

    @inbounds begin # Necessary for interpolation
        integ.k1 = k7; integ.k2 = k2; integ.k3 = k3
        integ.k4 = k4; integ.k5 = k5; integ.k6 = k6
        integ.k7 = k7
    end

    integ.tprev = t
    integ.t += dt

    return  nothing
end

# saveat is just a bool here:
#  true: ts is a vector of timestamps to read from
#  false: each ODE has its own timestamps, so ts is a vector to write to
function tsit5_kernel(_prob, ps, _us, _ts, dt,
                      ::Val{saveat}, ::Val{save_everystep}) where {saveat, save_everystep}
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    i >= length(ps) && return

    # get the actual problem for this thread
    p = @inbounds ps[i]
    prob = remake(_prob; p)

    # get the input/output arrays for this thread
    ts = if saveat
        _ts
    else
        @inbounds view(_ts, :, i)
    end
    us = @inbounds view(_us, :, i)
    
    integ = gputsit5_init(prob.f,false,prob.u0, prob.tspan[1], dt, prob.p)
    # TODO: optimize contiguous view to return a CuDeviceArray

    # FSAL
    for i in 2:length(ts)
        step!(integ)
        if saveat
            # TODO
        elseif save_everystep
            @inbounds us[i] = integ.u
            @inbounds ts[i] = integ.t
        end
    end

    if !saveat && !save_everystep
        @inbounds us[2] = integ.u
        @inbounds ts[2] = integ.t
    end

    return nothing
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

function atsit5_kernel(_prob, ps, _us, _ts, dt, abstol, reltol,
    ::Val{saveat}, ::Val{save_everystep}) where {saveat,save_everystep}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i >= length(ps) && return

    # get the actual problem for this thread
    p = @inbounds ps[i]
    prob = remake(_prob; p)

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

    u = u0
    k7 = f(u, p, t)

    cs, as, btildes, rs = SimpleDiffEq._build_atsit5_caches(eltype(u0))
    c1, c2, c3, c4, c5, c6 = cs
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54,
    a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76 = as
    btilde1, btilde2, btilde3, btilde4, btilde5, btilde6, btilde7 = btildes

    beta1, beta2, qmax, qmin, gamma, qoldinit, qold = build_adaptive_tsit5_controller_cache(eltype(u0))

    while t < tspan[2]
        uprev = u
        k1 = k7
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
                dtold = dt
                dt = dt / q #dtnew
                dt = min(abs(dt), abs(tf - t - dtold))
                told = t
                if (tf - t - dtold) < 1e-14
                    t = tf
                else
                    t += dtold
                end

                if saveat === nothing && save_everystep
                    error("Do not use saveat == nothing & save_everystep = true in adaptive version")
                else saveat !== nothing
                    while cur_t <= length(ts) && ts[cur_t] <= t
                        savet = ts[cur_t]
                        θ = (savet - told) / dtold
                        b1θ, b2θ, b3θ, b4θ, b5θ, b6θ, b7θ = SimpleDiffEq.bθs(rs, θ)
                        us[cur_t] = uprev + dtold * (
                            b1θ * k1 + b2θ * k2 + b3θ * k3 + b4θ * k4 + b5θ * k5 + b6θ * k6 + b7θ * k7)
                        cur_t += 1
                    end
                end

            end
        end
    end

    if !saveat && !save_everystep
        @inbounds us[2] = u
        @inbounds ts[2] = t
    end

    return nothing
end
