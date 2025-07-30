@inline function step!(integ::GPURodas4I{false, S, T}, ts, us) where {T, S}
    dt = integ.dt
    t = integ.t
    p = integ.p
    tmp = integ.tmp
    f = integ.f
    integ.uprev = integ.u
    uprev = integ.u
    @unpack a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, C21, C31, C32, C41, C42, C43,
    C51, C52, C53, C54, C61, C62, C63, C64, C65, γ, c2, c3, c4, d1, d2, d3, d4 = integ.tab

    integ.tprev = t
    saved_in_cb = false
    adv_integ = true
    ## Check if tstops are within the range of time-series
    if integ.tstops !== nothing && integ.tstops_idx <= length(integ.tstops) &&
       (integ.tstops[integ.tstops_idx] - integ.t - integ.dt - 100 * eps(T) < 0)
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
        @inbounds k1 = integ.k1
    end

    # Jacobian

    Jf, _ = build_J_W(integ.alg, f, γ, dt)
    J = Jf(uprev, p, t)

    Tgrad = build_tgrad(integ.alg, f)
    dT = Tgrad(uprev, p, t)

    # Precalculations
    dtC21 = C21 / dt
    dtC31 = C31 / dt
    dtC32 = C32 / dt
    dtC41 = C41 / dt
    dtC42 = C42 / dt
    dtC43 = C43 / dt
    dtC51 = C51 / dt
    dtC52 = C52 / dt
    dtC53 = C53 / dt
    dtC54 = C54 / dt
    dtC61 = C61 / dt
    dtC62 = C62 / dt
    dtC63 = C63 / dt
    dtC64 = C64 / dt
    dtC65 = C65 / dt

    dtd1 = dt * d1
    dtd2 = dt * d2
    dtd3 = dt * d3
    dtd4 = dt * d4
    dtgamma = dt * γ

    # Starting
    mass_matrix = f.mass_matrix
    W = mass_matrix / dtgamma - J
    du = f(uprev, p, t)

    # Step 1
    linsolve_tmp = du + dtd1 * dT
    k1 = linear_solve(W, -linsolve_tmp)
    u = uprev + a21 * k1
    du = f(u, p, t + c2 * dt)

    # Step 2
    linsolve_tmp = du + dtd2 * dT + dtC21 * k1
    k2 = linear_solve(W, -linsolve_tmp)
    u = uprev + a31 * k1 + a32 * k2
    du = f(u, p, t + c3 * dt)

    # Step 3
    linsolve_tmp = du + dtd3 * dT + (dtC31 * k1 + dtC32 * k2)
    k3 = linear_solve(W, -linsolve_tmp)
    u = uprev + a41 * k1 + a42 * k2 + a43 * k3
    du = f(u, p, t + c4 * dt)

    # Step 4
    linsolve_tmp = du + dtd4 * dT + (dtC41 * k1 + dtC42 * k2 + dtC43 * k3)
    k4 = linear_solve(W, -linsolve_tmp)
    u = uprev + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4
    du = f(u, p, t + dt)

    # Step 5
    linsolve_tmp = du + (dtC52 * k2 + dtC54 * k4 + dtC51 * k1 + dtC53 * k3)
    k5 = linear_solve(W, -linsolve_tmp)
    u = u + k5
    du = f(u, p, t + dt)

    # Step 6
    linsolve_tmp = du + (dtC61 * k1 + dtC62 * k2 + dtC65 * k5 + dtC64 * k4 + dtC63 * k3)
    k6 = linear_solve(W, -linsolve_tmp)
    integ.u = u + k6

    @inbounds begin # Necessary for interpolation
        @unpack h21, h22, h23, h24, h25, h31, h32, h33, h34, h35 = integ.tab
        integ.k1 = h21 * k1 + h22 * k2 + h23 * k3 + h24 * k4 + h25 * k5
        integ.k2 = h31 * k1 + h32 * k2 + h33 * k3 + h34 * k4 + h35 * k5
        # integ.k1 = k1
        # integ.k2 = k2
    end

    _, saved_in_cb = handle_callbacks!(integ, ts, us)

    return saved_in_cb
end

@inline function step!(integ::GPUARodas4I{false, S, T}, ts, us) where {T, S}
    beta1, beta2, qmax, qmin, gamma, qoldinit,
    _ = build_adaptive_controller_cache(
        integ.alg,
        T)

    dt = integ.dtnew
    t = integ.t
    p = integ.p
    tf = integ.tf

    tmp = integ.tmp
    f = integ.f
    integ.uprev = integ.u
    uprev = integ.u

    qold = integ.qold
    abstol = integ.abstol
    reltol = integ.reltol

    @unpack a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, C21, C31, C32, C41, C42, C43,
    C51, C52, C53, C54, C61, C62, C63, C64, C65, γ, c2, c3, c4, d1, d2, d3,
    d4 = integ.tab

    # Jacobian

    Jf, _ = build_J_W(integ.alg, f, γ, dt)
    J = Jf(uprev, p, t)

    Tgrad = build_tgrad(integ.alg, f)
    dT = Tgrad(uprev, p, t)

    if integ.u_modified
        k1 = f(uprev, p, t)
        integ.u_modified = false
    else
        @inbounds k1 = integ.k1
    end

    EEst = convert(T, Inf)

    while EEst > convert(T, 1.0)
        dt < convert(T, 1.0f-14) && error("dt<dtmin")

        # Precalculations
        dtC21 = C21 / dt
        dtC31 = C31 / dt
        dtC32 = C32 / dt
        dtC41 = C41 / dt
        dtC42 = C42 / dt
        dtC43 = C43 / dt
        dtC51 = C51 / dt
        dtC52 = C52 / dt
        dtC53 = C53 / dt
        dtC54 = C54 / dt
        dtC61 = C61 / dt
        dtC62 = C62 / dt
        dtC63 = C63 / dt
        dtC64 = C64 / dt
        dtC65 = C65 / dt

        dtd1 = dt * d1
        dtd2 = dt * d2
        dtd3 = dt * d3
        dtd4 = dt * d4
        dtgamma = dt * γ

        # Starting
        mass_matrix = f.mass_matrix
        W = mass_matrix / dtgamma - J
        du = f(uprev, p, t)

        # Step 1
        linsolve_tmp = du + dtd1 * dT
        k1 = linear_solve(W, -linsolve_tmp)
        u = uprev + a21 * k1
        du = f(u, p, t + c2 * dt)

        # Step 2
        linsolve_tmp = du + dtd2 * dT + dtC21 * k1
        k2 = linear_solve(W, -linsolve_tmp)
        u = uprev + a31 * k1 + a32 * k2
        du = f(u, p, t + c3 * dt)

        # Step 3
        linsolve_tmp = du + dtd3 * dT + (dtC31 * k1 + dtC32 * k2)
        k3 = linear_solve(W, -linsolve_tmp)
        u = uprev + a41 * k1 + a42 * k2 + a43 * k3
        du = f(u, p, t + c4 * dt)

        # Step 4
        linsolve_tmp = du + dtd4 * dT + (dtC41 * k1 + dtC42 * k2 + dtC43 * k3)
        k4 = linear_solve(W, -linsolve_tmp)
        u = uprev + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4
        du = f(u, p, t + dt)

        # Step 5
        linsolve_tmp = du + (dtC52 * k2 + dtC54 * k4 + dtC51 * k1 + dtC53 * k3)
        k5 = linear_solve(W, -linsolve_tmp)
        u = u + k5
        du = f(u, p, t + dt)

        # Step 6
        linsolve_tmp = du + (dtC61 * k1 + dtC62 * k2 + dtC65 * k5 + dtC64 * k4 + dtC63 * k3)
        k6 = linear_solve(W, -linsolve_tmp)
        u = u + k6

        tmp = k6 ./ (abstol .+ max.(abs.(uprev), abs.(u)) * reltol)
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
                @unpack h21, h22, h23, h24, h25, h31, h32, h33, h34, h35 = integ.tab
                integ.k1 = h21 * k1 + h22 * k2 + h23 * k3 + h24 * k4 + h25 * k5
                integ.k2 = h31 * k1 + h32 * k2 + h33 * k3 + h34 * k4 + h35 * k5
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
