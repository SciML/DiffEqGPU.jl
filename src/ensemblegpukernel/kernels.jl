
@kernel function ode_solve_kernel(@Const(probs), alg, _us, _ts, dt, callback,
        tstops, nsteps,
        saveat, ::Val{save_everystep}) where {save_everystep}
    i = @index(Global, Linear)

    # get the actual problem for this thread
    prob = @inbounds probs[i]

    # get the input/output arrays for this thread
    ts = @inbounds view(_ts, :, i)
    us = @inbounds view(_us, :, i)

    _saveat = get(prob.kwargs, :saveat, nothing)

    saveat = _saveat === nothing ? saveat : _saveat

    # Check if initialization is needed for DAEs
    u0, p_init,
    init_success = if SciMLBase.has_initialization_data(prob.f)
        # Perform initialization using SimpleNonlinearSolve compatible algorithm
        gpu_initialization_solve(prob, SimpleTrustRegion(), 1e-6, 1e-6)
    else
        prob.u0, prob.p, true
    end

    # Use initialized values
    integ = init(alg, prob.f, false, u0, prob.tspan[1], dt, p_init, tstops,
        callback, save_everystep, saveat)
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
        ## Interpolate to tf
        @inbounds us[end] = integ(tspan[2])
        @inbounds ts[end] = tspan[2]
    end

    if saveat === nothing && !save_everystep
        @inbounds us[2] = integ.u
        @inbounds ts[2] = integ.t
    end
end

@kernel function ode_asolve_kernel(@Const(probs), alg, _us, _ts, dt, callback, tstops,
        abstol, reltol,
        saveat,
        ::Val{save_everystep}) where {save_everystep}
    i = @index(Global, Linear)

    # get the actual problem for this thread
    prob = @inbounds probs[i]
    # get the input/output arrays for this thread
    ts = @inbounds view(_ts, :, i)
    us = @inbounds view(_us, :, i)
    # TODO: optimize contiguous view to return a CuDeviceArray

    _saveat = get(prob.kwargs, :saveat, nothing)

    saveat = _saveat === nothing ? saveat : _saveat

    # Check if initialization is needed for DAEs
    u0, p_init,
    init_success = if SciMLBase.has_initialization_data(prob.f)
        # Perform initialization using SimpleNonlinearSolve compatible algorithm
        gpu_initialization_solve(prob, SimpleTrustRegion(), abstol, reltol)
    else
        prob.u0, prob.p, true
    end

    tspan = prob.tspan
    f = prob.f
    p = p_init

    t = tspan[1]
    tf = prob.tspan[2]

    integ = init(alg, prob.f, false, u0, prob.tspan[1], prob.tspan[2], dt,
        p,
        abstol, reltol, DiffEqBase.ODE_DEFAULT_NORM, tstops, callback,
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
        ## Interpolate to tf
        @inbounds us[end] = integ(tspan[2])
        @inbounds ts[end] = tspan[2]
    end

    if saveat === nothing && !save_everystep
        @inbounds us[2] = integ.u
        @inbounds ts[2] = integ.t
    end
end
