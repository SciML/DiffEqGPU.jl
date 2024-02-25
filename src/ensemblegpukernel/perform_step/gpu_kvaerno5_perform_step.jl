@inline function step!(integ::GPUKvaerno5I{false, S, T}, ts, us) where {T, S}
    dt = integ.dt
    t = integ.t
    p = integ.p
    tmp = integ.tmp
    f = integ.f
    integ.uprev = integ.u
    uprev = integ.u
    @unpack γ, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a63, a64, a65, a71, a73, a74, a75, a76, c3, c4, c5, c6 = integ.tab
    @unpack btilde1, btilde3, btilde4, btilde5, btilde6, btilde7 = integ.tab
    @unpack α31, α32, α41, α42, α43, α51, α52, α53, α61, α62, α63 = integ.tab

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

    ## Build nlsolver

    nlsolver = build_nlsolver(integ.alg, integ.u, integ.p, integ.t, integ.dt, integ.f,
        integ.tab.γ,
        integ.tab.c3)

    ## Steps

    # FSAL Step 1

    z₁ = dt * k1

    ##### Step 2
    @set! nlsolver.z = z₁
    @set! nlsolver.tmp = uprev + γ * z₁
    @set! nlsolver.c = γ

    nlsolver = nlsolve(nlsolver, integ)

    z₂ = nlsolver.z

    ##### Step 3

    z₃ = α31 * z₁ + α32 * z₂

    @set! nlsolver.z = z₃
    @set! nlsolver.tmp = uprev + a31 * z₁ + a32 * z₂
    @set! nlsolver.c = c3

    nlsolver = nlsolve(nlsolver, integ)

    z₃ = nlsolver.z

    ################################## Solve Step 4

    z₄ = α41 * z₁ + α42 * z₂ + α43 * z₃

    @set! nlsolver.z = z₄
    @set! nlsolver.tmp = uprev + a41 * z₁ + a42 * z₂ + a43 * z₃
    @set! nlsolver.c = c4
    nlsolver = nlsolve(nlsolver, integ)
    z₄ = nlsolver.z

    z₅ = α51 * z₁ + α52 * z₂ + α53 * z₃

    @set! nlsolver.z = z₅
    @set! nlsolver.tmp = uprev + a51 * z₁ + a52 * z₂ + a53 * z₃ + a54 * z₄
    @set! nlsolver.c = c5

    nlsolver = nlsolve(nlsolver, integ)
    z₅ = nlsolver.z

    z₆ = α61 * z₁ + α62 * z₂ + α63 * z₃

    @set! nlsolver.z = z₆
    @set! nlsolver.tmp = uprev + a61 * z₁ + a63 * z₃ + a64 * z₄ + a65 * z₅
    @set! nlsolver.c = c6

    nlsolver = nlsolve(nlsolver, integ)
    z₆ = nlsolver.z

    z₇ = a61 * z₁ + a63 * z₃ + a64 * z₄ + a65 * z₅ + γ * z₆

    @set! nlsolver.z = z₇
    @set! nlsolver.tmp = uprev + a71 * z₁ + a73 * z₃ + a74 * z₄ + a75 * z₅ + a76 * z₆
    @set! nlsolver.c = one(nlsolver.c)

    nlsolver = nlsolve(nlsolver, integ)
    z₇ = nlsolver.z

    integ.u = nlsolver.tmp + γ * z₇

    k2 = z₇ ./ dt

    @inbounds begin # Necessary for interpolation
        integ.k1 = f(integ.u, p, t)
        integ.k2 = k2
    end

    _, saved_in_cb = handle_callbacks!(integ, ts, us)

    return saved_in_cb
end

@inline function step!(integ::GPUAKvaerno5I{false, S, T}, ts, us) where {T, S}
    beta1, beta2, qmax, qmin, gamma, qoldinit, _ = build_adaptive_controller_cache(
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

    @unpack γ, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a63, a64, a65, a71, a73, a74, a75, a76, c3, c4, c5, c6 = integ.tab
    @unpack btilde1, btilde3, btilde4, btilde5, btilde6, btilde7 = integ.tab
    @unpack α31, α32, α41, α42, α43, α51, α52, α53, α61, α62, α63 = integ.tab

    if integ.u_modified
        k1 = f(uprev, p, t)
        integ.u_modified = false
    else
        @inbounds k1 = integ.k1
    end

    EEst = convert(T, Inf)

    while EEst > convert(T, 1.0)
        dt < convert(T, 1.0f-14) && error("dt<dtmin")

        ## Steps

        nlsolver = build_nlsolver(integ.alg, integ.u, integ.p, integ.t, dt, integ.f,
            integ.tab.γ,
            integ.tab.c3)

        # FSAL Step 1

        k1 = f(uprev, p, t)

        z₁ = dt * k1

        ##### Step 2
        @set! nlsolver.z = z₁
        @set! nlsolver.tmp = uprev + γ * z₁
        @set! nlsolver.c = γ

        nlsolver = nlsolve(nlsolver, integ)
        z₂ = nlsolver.z

        ##### Step 3

        z₃ = α31 * z₁ + α32 * z₂

        @set! nlsolver.z = z₃
        @set! nlsolver.tmp = uprev + a31 * z₁ + a32 * z₂
        @set! nlsolver.c = c3

        nlsolver = nlsolve(nlsolver, integ)
        z₃ = nlsolver.z

        ################################## Solve Step 4

        z₄ = α41 * z₁ + α42 * z₂ + α43 * z₃

        @set! nlsolver.z = z₄
        @set! nlsolver.tmp = uprev + a41 * z₁ + a42 * z₂ + a43 * z₃
        @set! nlsolver.c = c4

        nlsolver = nlsolve(nlsolver, integ)
        z₄ = nlsolver.z

        ################################## Solve Step 5

        z₅ = α51 * z₁ + α52 * z₂ + α53 * z₃

        @set! nlsolver.z = z₅
        @set! nlsolver.tmp = uprev + a51 * z₁ + a52 * z₂ + a53 * z₃ + a54 * z₄
        @set! nlsolver.c = c5

        nlsolver = nlsolve(nlsolver, integ)
        z₅ = nlsolver.z

        ################################## Solve Step 6

        z₆ = α61 * z₁ + α62 * z₂ + α63 * z₃

        @set! nlsolver.z = z₆
        @set! nlsolver.tmp = uprev + a61 * z₁ + a63 * z₃ + a64 * z₄ + a65 * z₅
        @set! nlsolver.c = c6

        nlsolver = nlsolve(nlsolver, integ)
        z₆ = nlsolver.z

        ################################## Solve Step 7

        z₇ = a61 * z₁ + a63 * z₃ + a64 * z₄ + a65 * z₅ + γ * z₆

        @set! nlsolver.z = z₇
        @set! nlsolver.tmp = uprev + a71 * z₁ + a73 * z₃ + a74 * z₄ + a75 * z₅ + a76 * z₆
        @set! nlsolver.c = one(nlsolver.c)

        nlsolver = nlsolve(nlsolver, integ)
        z₇ = nlsolver.z

        u = nlsolver.tmp + γ * z₇

        k2 = z₇ ./ dt

        W_eval = nlsolver.W(nlsolver.tmp + nlsolver.γ * z₇, p, t + nlsolver.c * dt)

        err = linear_solve(W_eval,
            btilde1 * z₁ + btilde3 * z₃ + btilde4 * z₄ + btilde5 * z₅ + btilde6 * z₆ +
            btilde7 * z₇)

        tmp = (err) ./
              (abstol .+ max.(abs.(uprev), abs.(u)) * reltol)
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
