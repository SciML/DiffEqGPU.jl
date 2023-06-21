@inline function step!(integ::GPURB23I{false, S, T}, ts, us) where {T, S}
    dt = integ.dt
    t = integ.t
    p = integ.p
    tmp = integ.tmp
    f = integ.f
    integ.uprev = integ.u
    uprev = integ.u
    d = integ.d

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

    γ = dt * d

    dto2 = dt / 2
    dto6 = dt / 6

    Jf, _ = build_J_W(integ.alg, f, γ, dt)
    J = Jf(uprev, p, t)

    Tgrad = build_tgrad(integ.alg, f)
    dT = Tgrad(uprev, p, t)

    mass_matrix = integ.f.mass_matrix
    W = mass_matrix - γ * J
    W_fact = W

    # F = lu(W)
    F₀ = f(uprev, p, t)
    k1 = linear_solve(W_fact, F₀ + γ * dT)

    F₁ = f(uprev + dto2 * k1, p, t + dto2)

    if mass_matrix === I
        k2 = linear_solve(W_fact, (F₁ - k1) + k1)
    else
        k2 = linear_solve(W_fact, (F₁ - mass_matrix * k1) + k1)
    end

    integ.u = uprev + dt * k2

    @inbounds begin # Necessary for interpolation
        integ.k1 = k1
        integ.k2 = k2
    end

    _, saved_in_cb = handle_callbacks!(integ, ts, us)

    return saved_in_cb
end

#############################Adaptive Version#####################################

@inline function step!(integ::GPUARB23I{false, S, T}, ts, us) where {S, T}
    beta1, beta2, qmax, qmin, gamma, qoldinit, _ = build_adaptive_controller_cache(integ.alg,
        T)
    dt = integ.dtnew
    t = integ.t
    p = integ.p
    tf = integ.tf

    tmp = integ.tmp
    f = integ.f
    integ.uprev = integ.u
    uprev = integ.u
    d = integ.d

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

    mass_matrix = integ.f.mass_matrix

    while EEst > convert(T, 1.0)
        dt < convert(T, 1.0f-14) && error("dt<dtmin")

        γ = dt * d

        dto2 = dt / 2
        dto6 = dt / 6

        Jf, _ = build_J_W(integ.alg, f, γ, dt)
        J = Jf(uprev, p, t)

        Tgrad = build_tgrad(integ.alg, f)
        dT = Tgrad(uprev, p, t)

        W = mass_matrix - γ * J
        W_fact = W

        # F = lu(W)
        F₀ = f(uprev, p, t)
        k1 = linear_solve(W_fact, F₀ + γ * dT)

        F₁ = f(uprev + dto2 * k1, p, t + dto2)

        if mass_matrix === I
            k2 = linear_solve(W_fact, (F₁ - k1) + k1)
        else
            k2 = linear_solve(W_fact, (F₁ - mass_matrix * k1) + k1)
        end

        u = uprev + dt * k2

        e32 = T(6) + sqrt(T(2))
        F₂ = f(u, p, t + dt)

        if mass_matrix === I
            k3 = linear_solve(W_fact, F₂ - e32 * (k2 - F₁) - 2 * (k1 - F₀) + dt * dT)

        else
            k3 = linear_solve(W_fact,
                F₂ - mass_matrix * (e32 * k2 + 2 * k1) +
                e32 * F₁ + 2 * F₀ + dt * dT)
        end

        tmp = dto6 * (k1 - 2 * k2 + k3)
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
