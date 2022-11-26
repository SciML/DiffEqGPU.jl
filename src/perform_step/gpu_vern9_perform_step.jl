@inline function step!(integ::GPUVern9I{false, S, T}, ts, us) where {T, S}
    @unpack dt = integ
    t = integ.t
    p = integ.p
    @unpack c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, a0201, a0301, a0302,
    a0401, a0403, a0501, a0503, a0504, a0601, a0604, a0605, a0701, a0704, a0705, a0706,
    a0801, a0806, a0807, a0901, a0906, a0907, a0908, a1001, a1006, a1007, a1008, a1009,
    a1101, a1106, a1107, a1108, a1109, a1110, a1201, a1206, a1207, a1208, a1209, a1210,
    a1211, a1301, a1306, a1307, a1308, a1309, a1310, a1311, a1312, a1401, a1406, a1407,
    a1408, a1409, a1410, a1411, a1412, a1413, a1501, a1506, a1507, a1508, a1509, a1510,
    a1511, a1512, a1513, a1514, a1601, a1606, a1607, a1608, a1609, a1610, a1611, a1612,
    a1613, b1, b8, b9, b10, b11, b12, b13, b14, b15, btilde1, btilde8, btilde9, btilde10,
    btilde11, btilde12, btilde13, btilde14, btilde15, btilde16 = integ.tab

    tmp = integ.tmp
    f = integ.f
    integ.uprev = integ.u
    uprev = integ.u

    integ.tprev = t
    saved_in_cb = false
    adv_integ = true
    ## Check if tstops are within the range of time-series
    if integ.tstops !== nothing && integ.tstops_idx <= length(integ.tstops) &&
       (integ.tstops[integ.tstops_idx] - integ.t - integ.dt - 100 * eps(integ.t) < 0)
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
        @inbounds k1 = integ.k1
    end

    k1 = f(uprev, p, t)
    a = dt * a0201
    k2 = f(uprev + a * k1, p, t + c1 * dt)
    k3 = f(uprev + dt * (a0301 * k1 + a0302 * k2), p, t + c2 * dt)
    k4 = f(uprev + dt * (a0401 * k1 + a0403 * k3), p, t + c3 * dt)
    k5 = f(uprev + dt * (a0501 * k1 + a0503 * k3 + a0504 * k4), p, t + c4 * dt)
    k6 = f(uprev + dt * (a0601 * k1 + a0604 * k4 + a0605 * k5), p, t + c5 * dt)
    k7 = f(uprev + dt * (a0701 * k1 + a0704 * k4 + a0705 * k5 + a0706 * k6), p, t + c6 * dt)
    k8 = f(uprev + dt * (a0801 * k1 + a0806 * k6 + a0807 * k7), p, t + c7 * dt)
    k9 = f(uprev + dt * (a0901 * k1 + a0906 * k6 + a0907 * k7 + a0908 * k8), p, t + c8 * dt)
    k10 = f(uprev + dt * (a1001 * k1 + a1006 * k6 + a1007 * k7 + a1008 * k8 + a1009 * k9),
            p, t + c9 * dt)
    k11 = f(uprev +
            dt *
            (a1101 * k1 + a1106 * k6 + a1107 * k7 + a1108 * k8 + a1109 * k9 + a1110 * k10),
            p, t + c10 * dt)
    k12 = f(uprev +
            dt *
            (a1201 * k1 + a1206 * k6 + a1207 * k7 + a1208 * k8 + a1209 * k9 + a1210 * k10 +
             a1211 * k11), p, t + c11 * dt)
    k13 = f(uprev +
            dt *
            (a1301 * k1 + a1306 * k6 + a1307 * k7 + a1308 * k8 + a1309 * k9 + a1310 * k10 +
             a1311 * k11 + a1312 * k12), p, t + c12 * dt)
    k14 = f(uprev +
            dt *
            (a1401 * k1 + a1406 * k6 + a1407 * k7 + a1408 * k8 + a1409 * k9 + a1410 * k10 +
             a1411 * k11 + a1412 * k12 + a1413 * k13), p, t + c13 * dt)
    g15 = uprev +
          dt *
          (a1501 * k1 + a1506 * k6 + a1507 * k7 + a1508 * k8 + a1509 * k9 + a1510 * k10 +
           a1511 * k11 + a1512 * k12 + a1513 * k13 + a1514 * k14)
    g16 = uprev +
          dt *
          (a1601 * k1 + a1606 * k6 + a1607 * k7 + a1608 * k8 + a1609 * k9 + a1610 * k10 +
           a1611 * k11 + a1612 * k12 + a1613 * k13)
    k15 = f(g15, p, t + dt)
    k16 = f(g16, p, t + dt)

    integ.u = uprev +
              dt *
              (b1 * k1 + b8 * k8 + b9 * k9 + b10 * k10 + b11 * k11 + b12 * k12 + b13 * k13 +
               b14 * k14 + b15 * k15)

    @inbounds begin # Necessary for interpolation
        integ.k1 = k1
        integ.k2 = k8
        integ.k3 = k9
        integ.k4 = k10
        integ.k5 = k11
        integ.k6 = k12
        integ.k7 = k13
        integ.k8 = k14
        integ.k9 = k15
        integ.k10 = k16
        # integ.k17 = k17
        # integ.k18 = k18
        # integ.k19 = k19
        # integ.k20 = k20
    end

    if integ.cb !== nothing
        _, saved_in_cb = apply_discrete_callback!(integ, ts, us,
                                                  integ.cb.discrete_callbacks...)
    else
        saved_in_cb = false
    end

    return saved_in_cb
end

function vern9_kernel(probs, _us, _ts, dt, callback, tstops, nsteps,
                      saveat, ::Val{save_everystep}) where {save_everystep}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(probs) && return

    # get the actual problem for this thread
    prob = @inbounds probs[i]

    # get the input/output arrays for this thread
    ts = @inbounds view(_ts, :, i)
    us = @inbounds view(_us, :, i)

    integ = gpuvern9_init(prob.f, false, prob.u0, prob.tspan[1], dt, prob.p, tstops,
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
    while integ.t < tspan[2]
        saved_in_cb = step!(integ, ts, us)
        if saveat === nothing && save_everystep && !saved_in_cb
            @inbounds us[integ.step_idx] = integ.u
            @inbounds ts[integ.step_idx] = integ.t
            integ.step_idx += 1
        elseif saveat !== nothing
            while cur_t <= length(saveat) && saveat[cur_t] <= integ.t
                savet = saveat[cur_t]
                Θ = (savet - integ.tprev) / integ.dt
                @inbounds us[cur_t] = _ode_interpolant(Θ, integ.dt, integ.uprev, integ)
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

#############################Adaptive Version#####################################

@inline function step!(integ::GPUAVern9I{false, S, T}, ts, us) where {S, T}
    beta1, beta2, qmax, qmin, gamma, qoldinit, _ = build_adaptive_tsit5_controller_cache(eltype(integ.u))
    dt = integ.dtnew
    t = integ.t
    p = integ.p
    tf = integ.tf

    @unpack c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, a0201, a0301, a0302,
    a0401, a0403, a0501, a0503, a0504, a0601, a0604, a0605, a0701, a0704, a0705, a0706,
    a0801, a0806, a0807, a0901, a0906, a0907, a0908, a1001, a1006, a1007, a1008, a1009,
    a1101, a1106, a1107, a1108, a1109, a1110, a1201, a1206, a1207, a1208, a1209, a1210,
    a1211, a1301, a1306, a1307, a1308, a1309, a1310, a1311, a1312, a1401, a1406, a1407,
    a1408, a1409, a1410, a1411, a1412, a1413, a1501, a1506, a1507, a1508, a1509, a1510,
    a1511, a1512, a1513, a1514, a1601, a1606, a1607, a1608, a1609, a1610, a1611, a1612,
    a1613, b1, b8, b9, b10, b11, b12, b13, b14, b15, btilde1, btilde8, btilde9, btilde10,
    btilde11, btilde12, btilde13, btilde14, btilde15, btilde16 = integ.tab

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

    EEst = Inf

    while EEst > 1.0
        dt < 1e-14 && error("dt<dtmin")

        k1 = f(uprev, p, t)
        a = dt * a0201
        k2 = f(uprev + a * k1, p, t + c1 * dt)
        k3 = f(uprev + dt * (a0301 * k1 + a0302 * k2), p, t + c2 * dt)
        k4 = f(uprev + dt * (a0401 * k1 + a0403 * k3), p, t + c3 * dt)
        k5 = f(uprev + dt * (a0501 * k1 + a0503 * k3 + a0504 * k4), p, t + c4 * dt)
        k6 = f(uprev + dt * (a0601 * k1 + a0604 * k4 + a0605 * k5), p, t + c5 * dt)
        k7 = f(uprev + dt * (a0701 * k1 + a0704 * k4 + a0705 * k5 + a0706 * k6), p,
               t + c6 * dt)
        k8 = f(uprev + dt * (a0801 * k1 + a0806 * k6 + a0807 * k7), p, t + c7 * dt)
        k9 = f(uprev + dt * (a0901 * k1 + a0906 * k6 + a0907 * k7 + a0908 * k8), p,
               t + c8 * dt)
        k10 = f(uprev +
                dt * (a1001 * k1 + a1006 * k6 + a1007 * k7 + a1008 * k8 + a1009 * k9),
                p, t + c9 * dt)
        k11 = f(uprev +
                dt *
                (a1101 * k1 + a1106 * k6 + a1107 * k7 + a1108 * k8 + a1109 * k9 +
                 a1110 * k10),
                p, t + c10 * dt)
        k12 = f(uprev +
                dt *
                (a1201 * k1 + a1206 * k6 + a1207 * k7 + a1208 * k8 + a1209 * k9 +
                 a1210 * k10 +
                 a1211 * k11), p, t + c11 * dt)
        k13 = f(uprev +
                dt *
                (a1301 * k1 + a1306 * k6 + a1307 * k7 + a1308 * k8 + a1309 * k9 +
                 a1310 * k10 +
                 a1311 * k11 + a1312 * k12), p, t + c12 * dt)
        k14 = f(uprev +
                dt *
                (a1401 * k1 + a1406 * k6 + a1407 * k7 + a1408 * k8 + a1409 * k9 +
                 a1410 * k10 +
                 a1411 * k11 + a1412 * k12 + a1413 * k13), p, t + c13 * dt)
        g15 = uprev +
              dt *
              (a1501 * k1 + a1506 * k6 + a1507 * k7 + a1508 * k8 + a1509 * k9 +
               a1510 * k10 +
               a1511 * k11 + a1512 * k12 + a1513 * k13 + a1514 * k14)
        g16 = uprev +
              dt *
              (a1601 * k1 + a1606 * k6 + a1607 * k7 + a1608 * k8 + a1609 * k9 +
               a1610 * k10 +
               a1611 * k11 + a1612 * k12 + a1613 * k13)
        k15 = f(g15, p, t + dt)
        k16 = f(g16, p, t + dt)

        u = uprev +
            dt *
            (b1 * k1 + b8 * k8 + b9 * k9 + b10 * k10 + b11 * k11 + b12 * k12 + b13 * k13 +
             b14 * k14 + b15 * k15)

        tmp = dt * (btilde1 * k1 + btilde8 * k8 + btilde9 * k9 + btilde10 * k10 +
               btilde11 * k11 + btilde12 * k12 + btilde13 * k13 + btilde14 * k14 +
               btilde15 * k15 + btilde16 * k16)
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
                integ.k2 = k8
                integ.k3 = k9
                integ.k4 = k10
                integ.k5 = k11
                integ.k6 = k12
                integ.k7 = k13
                integ.k8 = k14
                integ.k9 = k15
                integ.k10 = k16
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
                   integ.tstops[integ.tstops_idx] - integ.t - integ.dt -
                   100 * eps(integ.t) < 0
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

function avern9_kernel(probs, _us, _ts, dt, callback, tstops, abstol, reltol,
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

    integ = gpuavern9_init(prob.f, false, prob.u0, prob.tspan[1], prob.tspan[2], dt, prob.p,
                           abstol, reltol, DiffEqBase.ODE_DEFAULT_NORM, tstops, callback)

    while integ.t < tspan[2]
        step!(integ, ts, us)
        if saveat === nothing && save_everystep
            error("Do not use saveat == nothing & save_everystep = true in adaptive version")
        elseif saveat !== nothing
            while cur_t <= length(saveat) && saveat[cur_t] <= integ.t
                savet = saveat[cur_t]
                Θ = (savet - integ.tprev) / integ.dt
                @inbounds us[cur_t] = _ode_interpolant(Θ, integ.dt, integ.uprev, integ)
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
