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
        integ.k1 = k1
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
function tsit5_kernel(probs, _us, _ts, dt, callback, tstops, nsteps,
                      saveat, ::Val{save_everystep}) where {save_everystep}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(probs) && return

    # get the actual problem for this thread
    prob = @inbounds probs[i]

    # get the input/output arrays for this thread
    ts = @inbounds view(_ts, :, i)
    us = @inbounds view(_us, :, i)

    integ = gputsit5_init(prob.f, false, prob.u0, prob.tspan[1], dt, prob.p, tstops,
                          callback, save_everystep)

    u0 = prob.u0
    tspan = prob.tspan

    cur_t = 0
    if saveat !== nothing
        cur_t = 1
        if prob.tspan[1] == ts[1]
            cur_t += 1
            @inbounds us[1] = u0
        end
    else
        @inbounds ts[integ.step_idx] = prob.tspan[1]
        @inbounds us[integ.step_idx] = prob.u0
    end

    integ.step_idx += 1
    # FSAL
    while integ.step_idx <= nsteps
        saved_in_cb = step!(integ, ts, us)
        if saveat === nothing && save_everystep & !saved_in_cb
            @inbounds us[integ.step_idx] = integ.u
            @inbounds ts[integ.step_idx] = integ.t
        elseif saveat !== nothing
            while cur_t <= length(saveat) && saveat[cur_t] <= integ.t
                savet = saveat[cur_t]
                θ = (savet - integ.tprev) / integ.dt
                b1θ, b2θ, b3θ, b4θ, b5θ, b6θ, b7θ = SimpleDiffEq.bθs(integ.rs, θ)
                @inbounds us[cur_t] = integ.uprev +
                                      integ.dt *
                                      (b1θ * integ.k1 + b2θ * integ.k2 + b3θ * integ.k3 +
                                       b4θ * integ.k4 + b5θ * integ.k5 + b6θ * integ.k6 +
                                       b7θ * integ.k7)
                @inbounds ts[cur_t] = savet
                cur_t += 1
            end
        end
        if !saved_in_cb
            integ.step_idx += 1
        end
    end

    if saveat === nothing && !save_everystep
        @inbounds us[2] = integ.u
        @inbounds ts[2] = integ.t
    end

    return nothing
end

#############################Adaptive Version#####################################

@inline function step!(integ::GPUAT5I{false, S, T}, ts, us) where {S, T}
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
                if integ.tstops !== nothing && integ.tstops_idx <= length(integ.tstops) &&
                   integ.tstops[integ.tstops_idx] <= integ.t + dt
                    integ.t = integ.tstops[integ.tstops_idx]
                    integ.tstops_idx += 1
                else
                    ##Advance the integrator
                    integ.t += dt
                end
            end
        end
    end
    if integ.cb !== nothing
        _, saved_in_cb = DiffEqBase.apply_discrete_callback!(integ,
                                                             integ.cb.discrete_callbacks...)
    else
        saved_in_cb = false
    end
    return nothing
end

function atsit5_kernel(probs, _us, _ts, dt, callback, tstops, abstol, reltol,
                       saveat, ::Val{save_everystep}) where {save_everystep}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(probs) && return

    # get the actual problem for this thread
    prob = @inbounds probs[i]
    # get the input/output arrays for this thread
    ts = @inbounds view(_ts, :, i)
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
                           abstol, reltol, DiffEqBase.ODE_DEFAULT_NORM, tstops, callback)

    while integ.t < tspan[2]
        step!(integ, ts, us)
        if saveat === nothing && save_everystep
            error("Do not use saveat == nothing & save_everystep = true in adaptive version")
        elseif saveat !== nothing
            while cur_t <= length(saveat) && saveat[cur_t] <= integ.t
                savet = saveat[cur_t]
                θ = (savet - integ.tprev) / integ.dt
                b1θ, b2θ, b3θ, b4θ, b5θ, b6θ, b7θ = SimpleDiffEq.bθs(integ.rs, θ)
                @inbounds us[cur_t] = integ.uprev +
                                      integ.dt *
                                      (b1θ * integ.k1 + b2θ * integ.k2 + b3θ * integ.k3 +
                                       b4θ * integ.k4 + b5θ * integ.k5 + b6θ * integ.k6 +
                                       b7θ * integ.k7)
                @inbounds ts[cur_t] = savet
                cur_t += 1
            end
        end
    end

    if saveat === nothing && !save_everystep
        @inbounds us[2] = integ.u
        @inbounds ts[2] = integ.t
    end

    return nothing
end
