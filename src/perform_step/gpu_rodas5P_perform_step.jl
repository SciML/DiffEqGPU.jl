@inline function step!(integ::GPURodas5PI{false, S, T}, ts, us) where {T, S}
    dt = integ.dt
    t = integ.t
    p = integ.p
    tmp = integ.tmp
    f = integ.f
    integ.uprev = integ.u
    uprev = integ.u
    @unpack a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65,
    C21, C31, C32, C41, C42, C43, C51, C52, C53, C54, C61, C62, C63, C64, C65, C71, C72, C73, C74, C75, C76,
    C81, C82, C83, C84, C85, C86, C87, γ, d1, d2, d3, d4, d5, c2, c3, c4, c5 = integ.tab

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
    J = f.jac(uprev, p, t)
    dT = f.tgrad(uprev, p, t)

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
    dtC71 = C71 / dt
    dtC72 = C72 / dt
    dtC73 = C73 / dt
    dtC74 = C74 / dt
    dtC75 = C75 / dt
    dtC76 = C76 / dt
    dtC81 = C81 / dt
    dtC82 = C82 / dt
    dtC83 = C83 / dt
    dtC84 = C84 / dt
    dtC85 = C85 / dt
    dtC86 = C86 / dt
    dtC87 = C87 / dt

    dtd1 = dt * d1
    dtd2 = dt * d2
    dtd3 = dt * d3
    dtd4 = dt * d4
    dtd5 = dt * d5
    dtgamma = dt * γ

    # Starting
    W = J - I * inv(dtgamma)
    du = f(uprev, p, t)

    # Step 1
    linsolve_tmp = du + dtd1 * dT
    k1 = W \ -linsolve_tmp
    u = uprev + a21 * k1
    du = f(u, p, t + c2 * dt)

    # Step 2
    linsolve_tmp = du + dtd2 * dT + dtC21 * k1
    k2 = W \ -linsolve_tmp
    u = uprev + a31 * k1 + a32 * k2
    du = f(u, p, t + c3 * dt)

    # Step 3
    linsolve_tmp = du + dtd3 * dT + (dtC31 * k1 + dtC32 * k2)
    k3 = W \ -linsolve_tmp
    u = uprev + a41 * k1 + a42 * k2 + a43 * k3
    du = f(u, p, t + c4 * dt)

    # Step 4
    linsolve_tmp = du + dtd4 * dT + (dtC41 * k1 + dtC42 * k2 + dtC43 * k3)
    k4 = W \ -linsolve_tmp
    u = uprev + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4
    du = f(u, p, t + c5 * dt)

    # Step 5
    linsolve_tmp = du + dtd5 * dT + (dtC52 * k2 + dtC54 * k4 + dtC51 * k1 + dtC53 * k3)
    k5 = W \ -linsolve_tmp
    u = uprev + a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5
    du = f(u, p, t + dt)

    # Step 6
    linsolve_tmp = du + (dtC61 * k1 + dtC62 * k2 + dtC63 * k3 + dtC64 * k4 + dtC65 * k5)
    k6 = W \ -linsolve_tmp
    u = u + k6
    du = f(u, p, t + dt)

    # Step 7
    linsolve_tmp = du + (dtC71 * k1 + dtC72 * k2 + dtC73 * k3 + dtC74 * k4 + dtC75 * k5 +
                    dtC76 * k6)
    k7 = W \ -linsolve_tmp
    u = u + k7
    du = f(u, p, t + dt)

    # Step 8
    linsolve_tmp = du + (dtC81 * k1 + dtC82 * k2 + dtC83 * k3 + dtC84 * k4 + dtC85 * k5 +
                    dtC86 * k6 + dtC87 * k7)
    k8 = W \ -linsolve_tmp
    integ.u = u + k8

    @inbounds begin # Necessary for interpolation
        @unpack h21, h22, h23, h24, h25, h26, h27, h28, h31, h32, h33, h34,
        h35, h36, h37, h38, h41, h42, h43, h44, h45, h46, h47, h48 = integ.tab

        integ.k1 = h21 * k1 + h22 * k2 + h23 * k3 + h24 * k4 + h25 * k5 + h26 * k6 +
                   h27 * k7 + h28 * k8
        integ.k2 = h31 * k1 + h32 * k2 + h33 * k3 + h34 * k4 + h35 * k5 + h36 * k6 +
                   h37 * k7 + h38 * k8
        integ.k3 = h41 * k1 + h42 * k2 + h43 * k3 + h44 * k4 + h45 * k5 + h46 * k6 +
                   h47 * k7 + h48 * k8
    end

    _, saved_in_cb = handle_callbacks!(integ, ts, us)

    return saved_in_cb
end

# saveat is just a bool here:
#  true: ts is a vector of timestamps to read from
#  false: each ODE has its own timestamps, so ts is a vector to write to
@kernel function ode_solve_kernel(@Const(probs), alg::GPURodas5P, _us, _ts, dt,
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

    integ = gpurodas5P_init(alg, prob.f, false, prob.u0, prob.tspan[1], dt, prob.p,
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

    # @print(typeof(integ))

    if saveat === nothing && !save_everystep
        @inbounds us[2] = integ.u
        @inbounds ts[2] = integ.t
    end
end

@inline function step!(integ::GPUARodas5PI{false, S, T}, ts, us) where {T, S}
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

    @unpack a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65,
    C21, C31, C32, C41, C42, C43, C51, C52, C53, C54, C61, C62, C63, C64, C65, C71, C72, C73, C74, C75, C76,
    C81, C82, C83, C84, C85, C86, C87, γ, d1, d2, d3, d4, d5, c2, c3, c4, c5 = integ.tab

    # Jacobian
    J = f.jac(uprev, p, t)
    dT = f.tgrad(uprev, p, t)

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
        dtC71 = C71 / dt
        dtC72 = C72 / dt
        dtC73 = C73 / dt
        dtC74 = C74 / dt
        dtC75 = C75 / dt
        dtC76 = C76 / dt
        dtC81 = C81 / dt
        dtC82 = C82 / dt
        dtC83 = C83 / dt
        dtC84 = C84 / dt
        dtC85 = C85 / dt
        dtC86 = C86 / dt
        dtC87 = C87 / dt

        dtd1 = dt * d1
        dtd2 = dt * d2
        dtd3 = dt * d3
        dtd4 = dt * d4
        dtd5 = dt * d5
        dtgamma = dt * γ

        # Starting
        W = J - I * inv(dtgamma)
        du = f(uprev, p, t)

        # Step 1
        linsolve_tmp = du + dtd1 * dT
        k1 = W \ -linsolve_tmp
        u = uprev + a21 * k1
        du = f(u, p, t + c2 * dt)

        # Step 2
        linsolve_tmp = du + dtd2 * dT + dtC21 * k1
        k2 = W \ -linsolve_tmp
        u = uprev + a31 * k1 + a32 * k2
        du = f(u, p, t + c3 * dt)

        # Step 3
        linsolve_tmp = du + dtd3 * dT + (dtC31 * k1 + dtC32 * k2)
        k3 = W \ -linsolve_tmp
        u = uprev + a41 * k1 + a42 * k2 + a43 * k3
        du = f(u, p, t + c4 * dt)

        # Step 4
        linsolve_tmp = du + dtd4 * dT + (dtC41 * k1 + dtC42 * k2 + dtC43 * k3)
        k4 = W \ -linsolve_tmp
        u = uprev + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4
        du = f(u, p, t + c5 * dt)

        # Step 5
        linsolve_tmp = du + dtd5 * dT + (dtC52 * k2 + dtC54 * k4 + dtC51 * k1 + dtC53 * k3)
        k5 = W \ -linsolve_tmp
        u = uprev + a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5
        du = f(u, p, t + dt)

        # Step 6
        linsolve_tmp = du + (dtC61 * k1 + dtC62 * k2 + dtC63 * k3 + dtC64 * k4 + dtC65 * k5)
        k6 = W \ -linsolve_tmp
        u = u + k6
        du = f(u, p, t + dt)

        # Step 7
        linsolve_tmp = du +
                       (dtC71 * k1 + dtC72 * k2 + dtC73 * k3 + dtC74 * k4 + dtC75 * k5 +
                        dtC76 * k6)
        k7 = W \ -linsolve_tmp
        u = u + k7
        du = f(u, p, t + dt)

        # Step 8
        linsolve_tmp = du +
                       (dtC81 * k1 + dtC82 * k2 + dtC83 * k3 + dtC84 * k4 + dtC85 * k5 +
                        dtC86 * k6 + dtC87 * k7)
        k8 = W \ -linsolve_tmp
        u = u + k8

        tmp = k8 ./ (abstol .+ max.(abs.(uprev), abs.(u)) * reltol)
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
                @unpack h21, h22, h23, h24, h25, h26, h27, h28, h31, h32, h33, h34,
                h35, h36, h37, h38, h41, h42, h43, h44, h45, h46, h47, h48 = integ.tab

                integ.k1 = h21 * k1 + h22 * k2 + h23 * k3 + h24 * k4 + h25 * k5 + h26 * k6 +
                           h27 * k7 + h28 * k8
                integ.k2 = h31 * k1 + h32 * k2 + h33 * k3 + h34 * k4 + h35 * k5 + h36 * k6 +
                           h37 * k7 + h38 * k8
                integ.k3 = h41 * k1 + h42 * k2 + h43 * k3 + h44 * k4 + h45 * k5 + h46 * k6 +
                           h47 * k7 + h48 * k8
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

@kernel function ode_asolve_kernel(probs, alg::GPURodas5P, _us, _ts, dt, callback,
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

    integ = gpuarodas5P_init(alg, prob.f, false, prob.u0, prob.tspan[1], prob.tspan[2],
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
    # @print("Hello\n")
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
