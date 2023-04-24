@inline function step!(integ::GPURodas4I{false, S, T}, ts, us) where {T, S}
    dt = integ.dt
    t = integ.t
    p = integ.p
    tmp = integ.tmp
    f = integ.f
    integ.uprev = integ.u
    uprev = integ.u
    @unpack a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, C21, C31, C32, C41, C42, C43, C51, C52, C53, C54, C61, C62, C63, C64, C65, gamma, c2, c3, c4, d1, d2, d3, d4 = integ.tab

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

    dtd1 = dt * d1
    dtd2 = dt * d2
    dtd3 = dt * d3
    dtd4 = dt * d4
    dtgamma = dt * gamma

    du = f(uprev, p, t)
    W =  J-I* inv(dtgamma)

    linsolve_tmp = du + dtd1 * dT
    k1 = W \ -linsolve_tmp
    u = uprev + a21 * k1
    du = f(u, p, t + c2 * dt)

    linsolve_tmp = du + dtd2 * dT + dtC21 * k1
    k2 = W \ -linsolve_tmp
    u = uprev + a31 * k1 + a32 * k2
    du = f(u, p, t + c3 * dt)

    linsolve_tmp = du + dtd3 * dT + (dtC31 * k1 + dtC32 * k2)
    k3 = W \ -linsolve_tmp
    u = uprev + a41 * k1 + a42 * k2 + a43 * k3
    du = f(u, p, t + c4 * dt)

    linsolve_tmp = du + dtd4 * dT + (dtC41 * k1 + dtC42 * k2 + dtC43 * k3)
    k4 = W \ -linsolve_tmp
    u = uprev + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4
    du = f(u, p, t + dt)

    linsolve_tmp = du + (dtC52 * k2 + dtC54 * k4 + dtC51 * k1 + dtC53 * k3)
    k5 = W \ -linsolve_tmp
    u = u + k5
    du = f(u, p, t + dt)

    linsolve_tmp = du + (dtC61 * k1 + dtC62 * k2 + dtC65 * k5 + dtC64 * k4 + dtC63 * k3)
    k6 = W \ -linsolve_tmp
    integ.u = u + k6

    @inbounds begin # Necessary for interpolation
        integ.k1 = k1
        integ.k2 = k2
    end

    _, saved_in_cb = handle_callbacks!(integ, ts, us)

    return saved_in_cb
end

# saveat is just a bool here:
#  true: ts is a vector of timestamps to read from
#  false: each ODE has its own timestamps, so ts is a vector to write to
@kernel function ode_solve_kernel(@Const(probs), alg::GPURodas4, _us, _ts, dt,
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

    integ = gpurodas4_init(alg, prob.f, false, prob.u0, prob.tspan[1], dt, prob.p,
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
