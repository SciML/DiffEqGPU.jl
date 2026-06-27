@inline function step!(integ::GPURodas3I{false, S, T}, ts, us) where {T, S}
    dt = integ.dt
    t = integ.t
    p = integ.p
    f = integ.f
    integ.uprev = integ.u
    uprev = integ.u
    @unpack a21, a31, a32, a41, a42, a43, C21, C31, C32, C41, C42, C43,
        γ, c2, c3, d1, d2, d3, d4 = integ.tab

    integ.tprev = t
    saved_in_cb = false
    if integ.tstops !== nothing && integ.tstops_idx <= length(integ.tstops) &&
            (integ.tstops[integ.tstops_idx] - integ.t - integ.dt - 100 * eps(T) < 0)
        integ.t = integ.tstops[integ.tstops_idx]
        dt = integ.t - integ.tprev
        integ.tstops_idx += 1
    else
        integ.t += dt
    end

    Jf, _ = build_J_W(integ.alg, f, γ, dt)
    J = Jf(uprev, p, t)

    Tgrad = build_tgrad(integ.alg, f)
    dT = Tgrad(uprev, p, t)

    dtC21 = C21 / dt
    dtC31 = C31 / dt
    dtC32 = C32 / dt
    dtC41 = C41 / dt
    dtC42 = C42 / dt
    dtC43 = C43 / dt

    dtd1 = dt * d1
    dtd2 = dt * d2
    dtd3 = dt * d3
    dtd4 = dt * d4
    dtgamma = dt * γ

    mass_matrix = f.mass_matrix
    W = J - mass_matrix * inv(dtgamma)
    du0 = f(uprev, p, t)

    linsolve_tmp = du0 + dtd1 * dT
    k1 = linear_solve(W, -linsolve_tmp)
    u = uprev + a21 * k1
    du = f(u, p, t + c2 * dt)

    linsolve_tmp = du + dtd2 * dT + mass_matrix * (dtC21 * k1)
    k2 = linear_solve(W, -linsolve_tmp)
    u = uprev + a31 * k1 + a32 * k2
    du = f(u, p, t + c3 * dt)

    linsolve_tmp = du + dtd3 * dT + mass_matrix * (dtC31 * k1 + dtC32 * k2)
    k3 = linear_solve(W, -linsolve_tmp)
    u = uprev + a41 * k1 + a42 * k2 + a43 * k3
    du = f(u, p, t + dt)

    linsolve_tmp = du + dtd4 * dT + mass_matrix * (dtC41 * k1 + dtC42 * k2 + dtC43 * k3)
    k4 = linear_solve(W, -linsolve_tmp)
    integ.u = u + k4

    @inbounds begin
        integ.k1 = du0
        integ.k2 = f(integ.u, p, t + dt)
    end

    _, saved_in_cb = handle_callbacks!(integ, ts, us)

    return saved_in_cb
end

@inline function step!(integ::GPUARodas3I{false, S, T}, ts, us) where {T, S}
    beta1, beta2, qmax, qmin, gamma, qoldinit,
        _ = build_adaptive_controller_cache(
        integ.alg,
        T
    )

    dt = integ.dtnew
    t = integ.t
    p = integ.p
    tf = integ.tf

    f = integ.f
    integ.uprev = integ.u
    uprev = integ.u

    qold = integ.qold
    abstol = integ.abstol
    reltol = integ.reltol

    @unpack a21, a31, a32, a41, a42, a43, C21, C31, C32, C41, C42, C43,
        γ, c2, c3, d1, d2, d3, d4 = integ.tab

    Jf, _ = build_J_W(integ.alg, f, γ, dt)
    J = Jf(uprev, p, t)

    Tgrad = build_tgrad(integ.alg, f)
    dT = Tgrad(uprev, p, t)

    EEst = convert(T, Inf)

    while EEst > convert(T, 1.0)
        dt < convert(T, 1.0f-14) && error("dt<dtmin")

        dtC21 = C21 / dt
        dtC31 = C31 / dt
        dtC32 = C32 / dt
        dtC41 = C41 / dt
        dtC42 = C42 / dt
        dtC43 = C43 / dt

        dtd1 = dt * d1
        dtd2 = dt * d2
        dtd3 = dt * d3
        dtd4 = dt * d4
        dtgamma = dt * γ

        mass_matrix = f.mass_matrix
        W = J - mass_matrix * inv(dtgamma)
        du0 = f(uprev, p, t)

        linsolve_tmp = du0 + dtd1 * dT
        k1 = linear_solve(W, -linsolve_tmp)
        u = uprev + a21 * k1
        du = f(u, p, t + c2 * dt)

        linsolve_tmp = du + dtd2 * dT + mass_matrix * (dtC21 * k1)
        k2 = linear_solve(W, -linsolve_tmp)
        u = uprev + a31 * k1 + a32 * k2
        du = f(u, p, t + c3 * dt)

        linsolve_tmp = du + dtd3 * dT + mass_matrix * (dtC31 * k1 + dtC32 * k2)
        k3 = linear_solve(W, -linsolve_tmp)
        u = uprev + a41 * k1 + a42 * k2 + a43 * k3
        du = f(u, p, t + dt)

        linsolve_tmp = du + dtd4 * dT + mass_matrix * (dtC41 * k1 + dtC42 * k2 + dtC43 * k3)
        k4 = linear_solve(W, -linsolve_tmp)
        u = u + k4

        tmp = (2 * k1 + k3) ./ (abstol .+ max.(abs.(uprev), abs.(u)) * reltol)
        EEst = DiffEqBase.ODE_DEFAULT_NORM(tmp, t)

        if iszero(EEst)
            q = inv(qmax)
        else
            q11 = EEst^beta1
            q = q11 / (qold^beta2)
        end

        if EEst > 1
            dt = dt / min(inv(qmin), q11 / gamma)
        else
            q = max(inv(qmax), min(inv(qmin), q / gamma))
            qold = max(EEst, qoldinit)
            dtnew = dt / q
            dtnew = min(abs(dtnew), abs(tf - t - dt))

            @inbounds begin
                integ.k1 = du0
                integ.k2 = f(u, p, t + dt)
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
                    integ.t += dt
                end
            end
        end
    end

    _, saved_in_cb = handle_callbacks!(integ, ts, us)

    return saved_in_cb
end
