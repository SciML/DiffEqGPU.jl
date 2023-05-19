@inline function step!(integ::GPUKvaerno3I{false, S, T}, ts, us) where {T, S}
    dt = integ.dt
    t = integ.t
    p = integ.p
    tmp = integ.tmp
    f = integ.f
    integ.uprev = integ.u
    uprev = integ.u
    @unpack γ, a31, a32, a41, a42, a43, btilde1, btilde2, btilde3, btilde4, c3, α31, α32 = integ.tab

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

    nlsolver = build_nlsolver(integ.u, integ.p, integ.t, integ.dt, integ.f, integ.tab.γ,
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

    @set! nlsolver.z = z₄ = a31 * z₁ + a32 * z₂ + γ * z₃ # use yhat as prediction
    @set! nlsolver.tmp = uprev + a41 * z₁ + a42 * z₂ + a43 * z₃
    @set! nlsolver.c = one(nlsolver.c)
    nlsolver = nlsolve(nlsolver, integ)
    z₄ = nlsolver.z

    integ.u = nlsolver.tmp + γ * z₄

    k2 = z₄ ./ dt

    @inbounds begin # Necessary for interpolation
        integ.k1 = f(integ.u, p, t)
        integ.k2 = k2
    end

    _, saved_in_cb = handle_callbacks!(integ, ts, us)

    return saved_in_cb
end

# saveat is just a bool here:
#  true: ts is a vector of timestamps to read from
#  false: each ODE has its own timestamps, so ts is a vector to write to
@kernel function ode_solve_kernel(@Const(probs), alg::GPUKvaerno3, _us, _ts, dt,
                                  callback, tstops, nsteps,
                                  saveat, ::Val{save_everystep}) where {save_everystep}
    i = @index(Global, Linear)

    # get the actual problem for this thread
    prob = @inbounds probs[i]

    # get the input/output arrays for this thread
    ts = @inbounds view(_ts, :, i)
    us = @inbounds view(_us, :, i)

    _saveat = get(prob.kwargs, :saveat, nothing)

    saveat = _saveat === nothing ? saveat : _saveat

    integ = gpukvaerno3_init(alg, prob.f, false, prob.u0, prob.tspan[1], dt, prob.p,
                             tstops,
                             callback, save_everystep, saveat)

    u0 = prob.u0
    tspan = prob.tspan

    integ.cur_t = 0
    if saveat !== nothing
        integ.cur_t = 1
        if prob.tspan[1] == saveat[1]
            integ.cur_t += 1
            @inbounds us[1] = u0
        end
    else
        @inbounds ts[integ.step_idx] = prob.tspan[1]
        @inbounds us[integ.step_idx] = prob.u0
    end

    integ.step_idx += 1

    # nlsolver = build_nlsolver(integ.u, integ.p, integ.t, integ.dt, integ.f, integ.tab.γ, integ.tab.c3)
    # FSAL

    while integ.t < tspan[2] && integ.retcode != DiffEqBase.ReturnCode.Terminated
        saved_in_cb = step!(integ, ts, us)
        !saved_in_cb && savevalues!(integ, ts, us)
    end
    if integ.t > tspan[2] && saveat === nothing
        ## Intepolate to tf
        @inbounds us[end] = integ(tspan[2])
        @inbounds ts[end] = tspan[2]
    end

    if saveat === nothing && !save_everystep
        @inbounds us[2] = integ.u
        @inbounds ts[2] = integ.t
    end
end

@inline function step!(integ::GPUAKvaerno3I{false, S, T}, ts, us) where {T, S}
    beta1, beta2, qmax, qmin, gamma, qoldinit, _ = build_adaptive_controller_cache(integ.alg,
                                                                                   eltype(integ.u))

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

    @unpack γ, a31, a32, a41, a42, a43, btilde1, btilde2, btilde3, btilde4, c3, α31, α32 = integ.tab

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

        nlsolver = build_nlsolver(integ.u, integ.p, integ.t, dt, integ.f, integ.tab.γ,
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

        @set! nlsolver.z = z₄ = a31 * z₁ + a32 * z₂ + γ * z₃ # use yhat as prediction
        @set! nlsolver.tmp = uprev + a41 * z₁ + a42 * z₂ + a43 * z₃
        @set! nlsolver.c = one(nlsolver.c)
        nlsolver = nlsolve(nlsolver, integ)
        z₄ = nlsolver.z

        u = nlsolver.tmp + γ * z₄

        k2 = z₄ ./ dt

        W_eval = nlsolver.W(nlsolver.tmp + nlsolver.γ * z₄, p, t + nlsolver.c * dt)

        err = W_eval \ (btilde1 * z₁ + btilde2 * z₂ + btilde3 * z₃ + btilde4 * z₄)

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

@kernel function ode_asolve_kernel(probs, alg::GPUKvaerno3, _us, _ts, dt, callback,
                                   tstops,
                                   abstol, reltol,
                                   saveat, ::Val{save_everystep}) where {save_everystep}
    i = @index(Global, Linear)

    # get the actual problem for this thread
    prob = @inbounds probs[i]
    # get the input/output arrays for this thread
    ts = @inbounds view(_ts, :, i)
    us = @inbounds view(_us, :, i)
    # TODO: optimize contiguous view to return a CuDeviceArray

    _saveat = get(prob.kwargs, :saveat, nothing)

    saveat = _saveat === nothing ? saveat : _saveat

    u0 = prob.u0
    tspan = prob.tspan
    f = prob.f
    p = prob.p

    t = tspan[1]
    tf = prob.tspan[2]

    integ = gpuakvaerno3_init(alg, prob.f, false, prob.u0, prob.tspan[1], prob.tspan[2],
                              dt, prob.p,
                              abstol, reltol, DiffEqBase.ODE_DEFAULT_NORM, tstops,
                              callback,
                              saveat)

    integ.cur_t = 0
    if saveat !== nothing
        integ.cur_t = 1
        if tspan[1] == saveat[1]
            integ.cur_t += 1
            @inbounds us[1] = u0
        end
    else
        @inbounds ts[1] = tspan[1]
        @inbounds us[1] = u0
    end

    while integ.t < tspan[2] && integ.retcode != DiffEqBase.ReturnCode.Terminated
        saved_in_cb = step!(integ, ts, us)
        !saved_in_cb && savevalues!(integ, ts, us)
    end

    if integ.t > tspan[2] && saveat === nothing
        ## Intepolate to tf
        @inbounds us[end] = integ(tspan[2])
        @inbounds ts[end] = tspan[2]
    end

    if saveat === nothing && !save_everystep
        @inbounds us[2] = integ.u
        @inbounds ts[2] = integ.t
    end
end
