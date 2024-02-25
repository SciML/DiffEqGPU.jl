@inline function step!(integ::GPUV7I{false, S, T}, ts, us) where {T, S}
    @unpack dt = integ
    t = integ.t
    p = integ.p
    @unpack c2, c3, c4, c5, c6, c7, c8, a021, a031, a032, a041, a043, a051, a053, a054,
    a061, a063, a064, a065, a071, a073, a074, a075, a076, a081, a083, a084,
    a085, a086, a087, a091, a093, a094, a095, a096, a097, a098, a101, a103,
    a104, a105, a106, a107, b1, b4, b5, b6, b7, b8, b9 = integ.tab

    tmp = integ.tmp
    f = integ.f
    integ.uprev = integ.u
    uprev = integ.u

    integ.tprev = t
    saved_in_cb = false
    adv_integ = true
    ## Check if tstops are within the range of time-series
    if integ.tstops !== nothing && integ.tstops_idx <= length(integ.tstops) &&
       (integ.tstops[integ.tstops_idx] - integ.t - integ.dt - 100 * eps(T) < 0)
        integ.t = integ.tstops[integ.tstops_idx]
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
        @inbounds k1 = integ.k1
    end

    k1 = f(uprev, p, t)
    a = dt * a021
    k2 = f(uprev + a * k1, p, t + c2 * dt)
    k3 = f(uprev + dt * (a031 * k1 + a032 * k2), p, t + c3 * dt)
    k4 = f(uprev + dt * (a041 * k1 + a043 * k3), p, t + c4 * dt)
    k5 = f(uprev + dt * (a051 * k1 + a053 * k3 + a054 * k4), p, t + c5 * dt)
    k6 = f(uprev + dt * (a061 * k1 + a063 * k3 + a064 * k4 + a065 * k5), p, t + c6 * dt)
    k7 = f(uprev + dt * (a071 * k1 + a073 * k3 + a074 * k4 + a075 * k5 + a076 * k6), p,
        t + c7 * dt)
    k8 = f(
        uprev +
        dt * (a081 * k1 + a083 * k3 + a084 * k4 + a085 * k5 + a086 * k6 + a087 * k7),
        p,
        t + c8 * dt)
    g9 = uprev +
         dt *
         (a091 * k1 + a093 * k3 + a094 * k4 + a095 * k5 + a096 * k6 + a097 * k7 + a098 * k8)
    g10 = uprev +
          dt * (a101 * k1 + a103 * k3 + a104 * k4 + a105 * k5 + a106 * k6 + a107 * k7)
    k9 = f(g9, p, t + dt)
    k10 = f(g10, p, t + dt)

    integ.u = uprev +
              dt * (b1 * k1 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7 + b8 * k8 + b9 * k9)

    @inbounds begin # Necessary for interpolation
        integ.k1 = k1
        integ.k2 = k2
        integ.k3 = k3
        integ.k4 = k4
        integ.k5 = k5
        integ.k6 = k6
        integ.k7 = k7
        integ.k8 = k8
        integ.k9 = k9
        integ.k10 = k10
    end

    _, saved_in_cb = handle_callbacks!(integ, ts, us)

    return saved_in_cb
end

#############################Adaptive Version#####################################

@inline function step!(integ::GPUAV7I{false, S, T}, ts, us) where {S, T}
    beta1, beta2, qmax, qmin, gamma, qoldinit, _ = build_adaptive_controller_cache(
        integ.alg,
        T)

    dt = integ.dtnew
    t = integ.t
    p = integ.p
    tf = integ.tf

    @unpack c2, c3, c4, c5, c6, c7, c8, a021, a031, a032, a041, a043, a051, a053, a054,
    a061, a063, a064, a065, a071, a073, a074, a075, a076, a081, a083, a084,
    a085, a086, a087, a091, a093, a094, a095, a096, a097, a098, a101, a103,
    a104, a105, a106, a107, b1, b4, b5, b6, b7, b8, b9, btilde1, btilde4,
    btilde5, btilde6, btilde7, btilde8, btilde9, btilde10, extra, interp = integ.tab

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
        @inbounds k1 = integ.k1
    end

    EEst = convert(T, Inf)

    while EEst > convert(T, 1.0)
        dt < convert(T, 1.0f-14) && error("dt<dtmin")

        k1 = f(uprev, p, t)
        a = dt * a021
        k2 = f(uprev + a * k1, p, t + c2 * dt)
        k3 = f(uprev + dt * (a031 * k1 + a032 * k2), p, t + c3 * dt)
        k4 = f(uprev + dt * (a041 * k1 + a043 * k3), p, t + c4 * dt)
        k5 = f(uprev + dt * (a051 * k1 + a053 * k3 + a054 * k4), p, t + c5 * dt)
        k6 = f(uprev + dt * (a061 * k1 + a063 * k3 + a064 * k4 + a065 * k5), p, t + c6 * dt)
        k7 = f(uprev + dt * (a071 * k1 + a073 * k3 + a074 * k4 + a075 * k5 + a076 * k6), p,
            t + c7 * dt)
        k8 = f(
            uprev +
            dt * (a081 * k1 + a083 * k3 + a084 * k4 + a085 * k5 + a086 * k6 + a087 * k7),
            p,
            t + c8 * dt)
        g9 = uprev +
             dt *
             (a091 * k1 + a093 * k3 + a094 * k4 + a095 * k5 + a096 * k6 + a097 * k7 +
              a098 * k8)
        g10 = uprev +
              dt * (a101 * k1 + a103 * k3 + a104 * k4 + a105 * k5 + a106 * k6 + a107 * k7)
        k9 = f(g9, p, t + dt)
        k10 = f(g10, p, t + dt)

        u = uprev +
            dt * (b1 * k1 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7 + b8 * k8 + b9 * k9)

        tmp = dt *
              (btilde1 * k1 + btilde4 * k4 + btilde5 * k5 + btilde6 * k6 + btilde7 * k7 +
               btilde8 * k8 + btilde9 * k9 + btilde10 * k10)

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
                integ.k8 = k8
                integ.k9 = k9
                integ.k10 = k10
            end

            integ.dt = dt
            integ.dtnew = dtnew
            integ.qold = qold
            integ.tprev = t
            integ.u = u

            if (tf - t - dt) < convert(T, 1.0f-14)
                integ.t = tf
            else
                if integ.tstops !== nothing && integ.tstops_idx <= length(integ.tstops) &&
                   integ.tstops[integ.tstops_idx] - integ.t - integ.dt -
                   100 * eps(T) < 0
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
