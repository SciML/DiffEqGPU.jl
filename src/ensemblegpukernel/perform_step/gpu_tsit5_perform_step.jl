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
       (integ.tstops[integ.tstops_idx] - integ.t - integ.dt - T(100) * eps(T) < T(0))
        integ.t = integ.tstops[integ.tstops_idx]
        ## Set correct dt
        dt = integ.t - integ.tprev
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

    _, saved_in_cb = handle_callbacks!(integ, ts, us)

    return saved_in_cb
end

#############################Adaptive Version#####################################

@inline function step!(integ::GPUAT5I{false, S, T}, ts, us) where {S, T}
    beta1, beta2, qmax, qmin, gamma, qoldinit,
    _ = build_adaptive_controller_cache(
        integ.alg,
        T)
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

    EEst = convert(T, Inf)

    while EEst > T(1.0)
        dt < T(1.0e-14) && error("dt<dtmin")

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

            if (tf - t - dt) < T(1.0e-14)
                integ.t = tf
            else
                if integ.tstops !== nothing && integ.tstops_idx <= length(integ.tstops) &&
                   integ.tstops[integ.tstops_idx] - integ.t - integ.dt -
                   T(100) * eps(T) < T(0)
                    integ.t = integ.tstops[integ.tstops_idx]
                    integ.u = integ(integ.t)
                    dt = integ.t - integ.tprev
                    integ.tstops_idx += 1
                else
                    ##Advance the integrator
                    integ.t += dt
                end
            end
        end
    end
    _, saved_in_cb = handle_callbacks!(integ, ts, us)

    return saved_in_cb
end