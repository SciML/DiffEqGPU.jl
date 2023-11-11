function build_adaptive_controller_cache(alg::A, ::Type{T}) where {A, T}
    beta1 = T(7 / (10 * alg_order(alg)))
    beta2 = T(2 / (5 * alg_order(alg)))
    qmax = T(10.0)
    qmin = T(1 / 5)
    gamma = T(9 / 10)
    qoldinit = T(1e-4)
    qold = qoldinit

    return beta1, beta2, qmax, qmin, gamma, qoldinit, qold
end

@inline function savevalues!(integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S,
            T,
        }, ts,
        us,
        force = false) where {AlgType <: GPUODEAlgorithm, IIP, S, T}
    saved, savedexactly = false, false

    saveat = integrator.saveat
    save_everystep = integrator.save_everystep

    if saveat === nothing && save_everystep
        saved = true
        savedexactly = true
        @inbounds us[integrator.step_idx] = integrator.u
        @inbounds ts[integrator.step_idx] = integrator.t
        integrator.step_idx += 1
    elseif saveat !== nothing
        saved = true
        savedexactly = true
        while integrator.cur_t <= length(saveat) && saveat[integrator.cur_t] <= integrator.t
            savet = saveat[integrator.cur_t]
            Θ = (savet - integrator.tprev) / integrator.dt
            @inbounds us[integrator.cur_t] = _ode_interpolant(Θ, integrator.dt,
                integrator.uprev, integrator)
            @inbounds ts[integrator.cur_t] = savet
            integrator.cur_t += 1
        end
    end

    saved, savedexactly
end

@inline function DiffEqBase.terminate!(integrator::DiffEqBase.AbstractODEIntegrator{AlgType,
            IIP, S,
            T},
        retcode = ReturnCode.Terminated) where {
        AlgType <:
        GPUODEAlgorithm,
        IIP,
        S,
        T,
    }
    integrator.retcode = retcode
end

@inline function apply_discrete_callback!(integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S, T,
        },
        ts, us,
        callback::GPUDiscreteCallback) where {
        AlgType <:
        GPUODEAlgorithm,
        IIP, S, T}
    saved_in_cb = false
    if callback.condition(integrator.u, integrator.t, integrator)
        # handle saveat
        _, savedexactly = savevalues!(integrator, ts, us)
        saved_in_cb = true
        integrator.u_modified = true
        callback.affect!(integrator)
    end
    integrator.u_modified, saved_in_cb
end

@inline function apply_discrete_callback!(integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S, T,
        },
        ts, us,
        callback::GPUDiscreteCallback,
        args...) where {AlgType <: GPUODEAlgorithm, IIP,
        S, T}
    apply_discrete_callback!(integrator, ts, us,
        apply_discrete_callback!(integrator, ts, us, callback)...,
        args...)
end

@inline function apply_discrete_callback!(integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S, T,
        },
        ts, us,
        discrete_modified::Bool,
        saved_in_cb::Bool, callback::GPUDiscreteCallback,
        args...) where {AlgType <: GPUODEAlgorithm, IIP,
        S, T}
    bool, saved_in_cb2 = apply_discrete_callback!(integrator, ts, us,
        apply_discrete_callback!(integrator, ts,
            us, callback)...,
        args...)
    discrete_modified || bool, saved_in_cb || saved_in_cb2
end

@inline function apply_discrete_callback!(integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S, T,
        },
        ts, us,
        discrete_modified::Bool,
        saved_in_cb::Bool,
        callback::GPUDiscreteCallback) where {
        AlgType <:
        GPUODEAlgorithm,
        IIP, S, T}
    bool, saved_in_cb2 = apply_discrete_callback!(integrator, ts, us, callback)
    discrete_modified || bool, saved_in_cb || saved_in_cb2
end

@inline function interpolate(integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S,
            T,
        },
        t) where {AlgType <: GPUODEAlgorithm, IIP, S, T}
    θ = (t - integrator.tprev) / integrator.dt
    b1θ, b2θ, b3θ, b4θ, b5θ, b6θ, b7θ = SimpleDiffEq.bθs(integrator.rs, θ)
    return integrator.uprev +
           integrator.dt *
           (b1θ * integrator.k1 + b2θ * integrator.k2 + b3θ * integrator.k3 +
            b4θ * integrator.k4 + b5θ * integrator.k5 + b6θ * integrator.k6 +
            b7θ * integrator.k7)
end

@inline function _change_t_via_interpolation!(integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S,
            T,
        },
        t,
        modify_save_endpoint::Type{Val{T1}}) where {
        AlgType <:
        GPUODEAlgorithm,
        IIP,
        S,
        T,
        T1,
    }
    # Can get rid of an allocation here with a function
    # get_tmp_arr(integrator.cache) which gives a pointer to some
    # cache array which can be modified.
    if integrator.tdir * t < integrator.tdir * integrator.tprev
        error("Current interpolant only works between tprev and t")
    elseif t != integrator.t
        integrator.u = integrator(t)
        integrator.step_idx -= Int(round((integrator.t - t) / integrator.dt))
        integrator.t = t
        #integrator.dt = integrator.t - integrator.tprev
    end
end
@inline function DiffEqBase.change_t_via_interpolation!(integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S,
            T,
        },
        t,
        modify_save_endpoint::Type{Val{T1}} = Val{
            false,
        }) where {
        AlgType <:
        GPUODEAlgorithm,
        IIP,
        S,
        T,
        T1,
    }
    _change_t_via_interpolation!(integrator, t, modify_save_endpoint)
end

@inline function apply_callback!(integrator::DiffEqBase.AbstractODEIntegrator{AlgType, IIP,
            S, T},
        callback::GPUContinuousCallback,
        cb_time, prev_sign, event_idx, ts,
        us) where {AlgType <: GPUODEAlgorithm, IIP, S, T}
    DiffEqBase.change_t_via_interpolation!(integrator, integrator.tprev + cb_time)

    # handle saveat
    _, savedexactly = savevalues!(integrator, ts, us)
    saved_in_cb = true

    integrator.u_modified = true

    if prev_sign < 0
        if callback.affect! === nothing
            integrator.u_modified = false
        else
            callback.affect!(integrator)
        end
    elseif prev_sign > 0
        if callback.affect_neg! === nothing
            integrator.u_modified = false
        else
            callback.affect_neg!(integrator)
        end
    end

    true, saved_in_cb
end

@inline function handle_callbacks!(integrator::DiffEqBase.AbstractODEIntegrator{AlgType,
            IIP, S, T},
        ts, us) where {AlgType <: GPUODEAlgorithm, IIP, S, T}
    discrete_callbacks = integrator.callback.discrete_callbacks
    continuous_callbacks = integrator.callback.continuous_callbacks
    atleast_one_callback = false

    continuous_modified = false
    discrete_modified = false
    saved_in_cb = false
    if !(continuous_callbacks isa Tuple{})
        event_occurred = false

        time, upcrossing, event_occurred, event_idx, idx, counter = DiffEqBase.find_first_continuous_callback(integrator,
            continuous_callbacks...)

        if event_occurred
            integrator.event_last_time = idx
            integrator.vector_event_last_time = event_idx
            continuous_modified, saved_in_cb = apply_callback!(integrator,
                continuous_callbacks[1],
                time, upcrossing,
                event_idx, ts, us)
        else
            integrator.event_last_time = 0
            integrator.vector_event_last_time = 1
        end
    end
    if !(discrete_callbacks isa Tuple{})
        discrete_modified, saved_in_cb = apply_discrete_callback!(integrator, ts, us,
            discrete_callbacks...)
        return discrete_modified, saved_in_cb
    end

    return false, saved_in_cb
end

@inline function DiffEqBase.find_callback_time(integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S,
            T,
        },
        callback::DiffEqGPU.GPUContinuousCallback,
        counter) where {AlgType <: GPUODEAlgorithm,
        IIP, S, T}
    event_occurred, interp_index, prev_sign, prev_sign_index, event_idx = DiffEqBase.determine_event_occurance(integrator,
        callback,
        counter)

    if event_occurred
        if callback.condition === nothing
            new_t = zero(typeof(integrator.t))
        else
            top_t = integrator.t
            bottom_t = integrator.tprev
            if callback.rootfind != SciMLBase.NoRootFind
                function zero_func(abst, p = nothing)
                    DiffEqBase.get_condition(integrator, callback, abst)
                end
                if zero_func(top_t) == 0
                    Θ = top_t
                else
                    if integrator.event_last_time == counter &&
                       abs(zero_func(bottom_t)) <= 100abs(integrator.last_event_error) &&
                       prev_sign_index == 1

                        # Determined that there is an event by derivative
                        # But floating point error may make the end point negative
                        bottom_t += integrator.dt * callback.repeat_nudge
                        sign_top = sign(zero_func(top_t))
                        sign(zero_func(bottom_t)) * sign_top >= zero(sign_top) &&
                            error("Double callback crossing floating pointer reducer errored. Report this issue.")
                    end
                    Θ = DiffEqBase.bisection(zero_func, (bottom_t, top_t),
                        isone(integrator.tdir),
                        callback.rootfind, callback.abstol,
                        callback.reltol)
                    integrator.last_event_error = DiffEqBase.ODE_DEFAULT_NORM(zero_func(Θ),
                        Θ)
                end
                new_t = Θ - integrator.tprev
            else
                # If no solve and no interpolants, just use endpoint
                new_t = integrator.dt
            end
        end
    else
        new_t = zero(typeof(integrator.t))
    end

    new_t, prev_sign, event_occurred, event_idx
end

@inline function SciMLBase.get_tmp_cache(integrator::DiffEqBase.AbstractODEIntegrator{
        AlgType,
        IIP,
        S, T}) where {
        AlgType <:
        GPUODEAlgorithm,
        IIP,
        S,
        T,
    }
    return nothing
end

@inline function DiffEqBase.get_condition(integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S, T,
        },
        callback,
        abst) where {AlgType <: GPUODEAlgorithm, IIP, S, T
    }
    if abst == integrator.t
        tmp = integrator.u
    elseif abst == integrator.tprev
        tmp = integrator.uprev
    else
        tmp = integrator(abst)
    end
    return callback.condition(tmp, abst, integrator)
end

# interp_points = 0 or equivalently nothing
@inline function DiffEqBase.determine_event_occurance(integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S,
            T,
        },
        callback::DiffEqGPU.GPUContinuousCallback,
        counter) where {
        AlgType <:
        GPUODEAlgorithm, IIP,
        S, T}
    event_occurred = false
    interp_index = 0

    # Check if the event occured
    previous_condition = callback.condition(integrator.uprev, integrator.tprev,
        integrator)

    prev_sign = zero(integrator.t)
    next_sign = zero(integrator.t)
    # @show typeof(0)
    if integrator.event_last_time == counter &&
       minimum(DiffEqBase.ODE_DEFAULT_NORM(previous_condition, integrator.t)) <=
       100DiffEqBase.ODE_DEFAULT_NORM(integrator.last_event_error, integrator.t)

        # If there was a previous event, utilize the derivative at the start to
        # chose the previous sign. If the derivative is positive at tprev, then
        # we treat `prev_sign` as negetive, and if the derivative is negative then we
        # treat `prev_sign` as positive, regardless of the postiivity/negativity
        # of the true value due to it being =0 sans floating point issues.

        # Only due this if the discontinuity did not move it far away from an event
        # Since near even we use direction instead of location to reset

        # Evaluate condition slightly in future
        abst = integrator.tprev + integrator.dt * callback.repeat_nudge
        tmp_condition = DiffEqBase.get_condition(integrator, callback, abst)
        prev_sign = sign(tmp_condition)
    else
        prev_sign = sign(previous_condition)
    end

    prev_sign_index = 1
    abst = integrator.t
    next_condition = DiffEqBase.get_condition(integrator, callback, abst)
    next_sign = sign(next_condition)

    if ((prev_sign < 0 && callback.affect! !== nothing) ||
        (prev_sign > 0 && callback.affect_neg! !== nothing)) && prev_sign * next_sign <= 0
        event_occurred = true
    end
    event_idx = 1

    event_occurred, interp_index, prev_sign, prev_sign_index, event_idx
end
