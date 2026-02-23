function build_adaptive_controller_cache(alg::A, ::Type{T}) where {A, T}
    beta1 = T(7 / (10 * alg_order(alg)))
    beta2 = T(2 / (5 * alg_order(alg)))
    qmax = T(10.0)
    qmin = T(1 / 5)
    gamma = T(9 / 10)
    qoldinit = T(1.0e-4)
    qold = qoldinit

    return beta1, beta2, qmax, qmin, gamma, qoldinit, qold
end

@inline function savevalues!(
        integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S,
            T,
        }, ts,
        us,
        force = false
    ) where {AlgType <: GPUODEAlgorithm, IIP, S, T}
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
            @inbounds us[integrator.cur_t] = _ode_interpolant(
                Θ, integrator.dt,
                integrator.uprev, integrator
            )
            @inbounds ts[integrator.cur_t] = savet
            integrator.cur_t += 1
        end
    end

    return saved, savedexactly
end

@inline function DiffEqBase.terminate!(
        integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP, S,
            T,
        },
        retcode = ReturnCode.Terminated
    ) where {
        AlgType <:
        GPUODEAlgorithm,
        IIP,
        S,
        T,
    }
    return integrator.retcode = retcode
end

@inline function apply_discrete_callback!(
        integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S, T,
        },
        ts, us,
        callback::GPUDiscreteCallback
    ) where {
        AlgType <:
        GPUODEAlgorithm,
        IIP, S, T,
    }
    saved_in_cb = false
    if callback.condition(integrator.u, integrator.t, integrator)
        # handle saveat
        _, savedexactly = savevalues!(integrator, ts, us)
        saved_in_cb = true
        integrator.u_modified = true
        callback.affect!(integrator)
    end
    return integrator.u_modified, saved_in_cb
end

@inline function apply_discrete_callback!(
        integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S, T,
        },
        ts, us,
        callback::GPUDiscreteCallback,
        args...
    ) where {
        AlgType <: GPUODEAlgorithm, IIP,
        S, T,
    }
    return apply_discrete_callback!(
        integrator, ts, us,
        apply_discrete_callback!(integrator, ts, us, callback)...,
        args...
    )
end

@inline function apply_discrete_callback!(
        integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S, T,
        },
        ts, us,
        discrete_modified::Bool,
        saved_in_cb::Bool, callback::GPUDiscreteCallback,
        args...
    ) where {
        AlgType <: GPUODEAlgorithm, IIP,
        S, T,
    }
    bool,
        saved_in_cb2 = apply_discrete_callback!(
        integrator, ts, us,
        apply_discrete_callback!(
            integrator, ts,
            us, callback
        )...,
        args...
    )
    return discrete_modified || bool, saved_in_cb || saved_in_cb2
end

@inline function apply_discrete_callback!(
        integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S, T,
        },
        ts, us,
        discrete_modified::Bool,
        saved_in_cb::Bool,
        callback::GPUDiscreteCallback
    ) where {
        AlgType <:
        GPUODEAlgorithm,
        IIP, S, T,
    }
    bool, saved_in_cb2 = apply_discrete_callback!(integrator, ts, us, callback)
    return discrete_modified || bool, saved_in_cb || saved_in_cb2
end

@inline function interpolate(
        integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S,
            T,
        },
        t
    ) where {AlgType <: GPUODEAlgorithm, IIP, S, T}
    θ = (t - integrator.tprev) / integrator.dt
    b1θ, b2θ, b3θ, b4θ, b5θ, b6θ, b7θ = SimpleDiffEq.bθs(integrator.rs, θ)
    return integrator.uprev +
        integrator.dt *
        (
        b1θ * integrator.k1 + b2θ * integrator.k2 + b3θ * integrator.k3 +
            b4θ * integrator.k4 + b5θ * integrator.k5 + b6θ * integrator.k6 +
            b7θ * integrator.k7
    )
end

@inline function _change_t_via_interpolation!(
        integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S,
            T,
        },
        t,
        modify_save_endpoint::Type{Val{T1}}
    ) where {
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
    return if integrator.tdir * t < integrator.tdir * integrator.tprev
        error("Current interpolant only works between tprev and t")
    elseif t != integrator.t
        integrator.u = integrator(t)
        integrator.step_idx -= Int(round((integrator.t - t) / integrator.dt))
        integrator.t = t
        #integrator.dt = integrator.t - integrator.tprev
    end
end
@inline function DiffEqBase.change_t_via_interpolation!(
        integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S,
            T,
        },
        t,
        modify_save_endpoint::Type{Val{T1}} = Val{
            false,
        }
    ) where {
        AlgType <:
        GPUODEAlgorithm,
        IIP,
        S,
        T,
        T1,
    }
    return _change_t_via_interpolation!(integrator, t, modify_save_endpoint)
end

@inline function apply_callback!(
        integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType, IIP,
            S, T,
        },
        callback::GPUContinuousCallback,
        cb_time, prev_sign, event_idx, ts,
        us
    ) where {AlgType <: GPUODEAlgorithm, IIP, S, T}
    DiffEqBase.change_t_via_interpolation!(integrator, cb_time)

    # The new absolute-time callback handling can leave dtnew ≈ 0 when
    # the step controller clamped it at tf before the callback moved t back.
    if hasfield(typeof(integrator), :dtnew) &&
            integrator.dtnew < convert(T, 1.0e-12)
        remaining = abs(integrator.tf - integrator.t)
        integrator.dtnew = min(
            callback.dtrelax * integrator.dt,
            remaining
        )
    end

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

    return true, saved_in_cb
end

@inline function handle_callbacks!(
        integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP, S, T,
        },
        ts, us
    ) where {AlgType <: GPUODEAlgorithm, IIP, S, T}
    discrete_callbacks = integrator.callback.discrete_callbacks
    continuous_callbacks = integrator.callback.continuous_callbacks
    atleast_one_callback = false

    continuous_modified = false
    discrete_modified = false
    saved_in_cb = false
    if !(continuous_callbacks isa Tuple{})
        event_occurred = false

        time, upcrossing,
            event_occurred,
            event_idx,
            idx,
            counter = DiffEqBase.find_first_continuous_callback(
            integrator,
            continuous_callbacks...
        )

        if event_occurred
            integrator.event_last_time = idx
            integrator.vector_event_last_time = event_idx
            continuous_modified,
                saved_in_cb = apply_callback!(
                integrator,
                continuous_callbacks[1],
                time, upcrossing,
                event_idx, ts, us
            )
        else
            integrator.event_last_time = 0
            integrator.vector_event_last_time = 1
        end
    end
    if !(discrete_callbacks isa Tuple{})
        discrete_modified,
            saved_in_cb = apply_discrete_callback!(
            integrator, ts, us,
            discrete_callbacks...
        )
        return discrete_modified, saved_in_cb
    end

    return false, saved_in_cb
end

@inline function gpu_find_root(f, tup, rootfind)
    # Hand-written ITP (Interpolate, Truncate, Project) root finder.
    # GPU-compatible: no dynamic dispatch, no allocations, no SciMLBase wrappers.
    # Algorithm matches BracketingNonlinearSolve.ITP() defaults (scaled_k1=0.2, k2=2, n0=10).
    left, right = tup
    fl = f(left)
    fr = f(right)

    T = typeof(left)
    span0 = right - left
    k1 = T(0.2) / span0              # scaled_k1 * span0^(1-k2), k2=2
    ϵ_s = span0 * T(512)             # span0/2 * 2^n0, n0=10
    T0 = zero(fl)

    for _ in 1:100
        span = right - left
        mid = (left + right) / 2
        r = ϵ_s - span / 2

        # Interpolate: regula falsi
        x_f = left + span * fl / (fl - fr)

        # Truncate: limit step toward bisection midpoint
        δ = max(k1 * span * span, eps(x_f))
        diff = mid - x_f
        xt = ifelse(δ ≤ abs(diff), x_f + copysign(δ, diff), mid)

        # Project: keep within trust region around midpoint
        xp = ifelse(abs(xt - mid) ≤ r, xt, mid - copysign(r, diff))

        # Evaluate and update bracket
        yp = f(xp)
        yps = yp * sign(fr)
        if yps > T0
            right = xp
            fr = yp
        elseif yps < T0
            left = xp
            fl = yp
        else
            left = xp
            right = xp
            break
        end

        ϵ_s /= 2

        if nextfloat(left) ≥ right
            break
        end
    end

    if rootfind == SciMLBase.LeftRootFind
        return left
    else
        return right
    end
end

@inline function DiffEqBase.find_callback_time(
        integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S,
            T,
        },
        callback::DiffEqGPU.GPUContinuousCallback,
        callback_idx
    ) where {
        AlgType <: GPUODEAlgorithm,
        IIP, S, T,
    }
    # Compute previous sign
    bottom_t = integrator.tprev
    bottom_condition = callback.condition(
        integrator.uprev, integrator.tprev,
        integrator
    )

    if integrator.event_last_time == callback_idx
        # If there was a previous event, nudge tprev on the right
        # side of the root (if necessary) to avoid repeat detection
        if abs(bottom_condition - integrator.last_event_error) <= callback.abstol
            bottom_t = integrator.tprev + integrator.dt * callback.repeat_nudge
            bottom_condition = DiffEqBase.get_condition(integrator, callback, bottom_t)
        end
    end
    bottom_sign = sign(bottom_condition)

    # Check if an event occurred
    top_t = integrator.t
    top_condition = DiffEqBase.get_condition(integrator, callback, top_t)
    top_sign = sign(top_condition)

    event_occurred = (
        (bottom_sign < 0 && callback.affect! !== nothing) ||
            (bottom_sign > 0 && callback.affect_neg! !== nothing)
    ) &&
        bottom_sign * top_sign <= 0

    event_idx = 1

    if !event_occurred
        callback_t = integrator.t
        residual = zero(bottom_condition)
    elseif callback.rootfind == SciMLBase.NoRootFind || iszero(top_sign)
        callback_t = top_t
        residual = zero(bottom_condition)
    else
        zero_func(abst, p = nothing) = DiffEqBase.get_condition(integrator, callback, abst)
        callback_t = gpu_find_root(zero_func, (bottom_t, top_t), callback.rootfind)
        residual = zero_func(callback_t)
    end

    return callback_t, bottom_sign, event_occurred, event_idx, residual
end

@inline function SciMLBase.get_tmp_cache(
        integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S, T,
        }
    ) where {
        AlgType <:
        GPUODEAlgorithm,
        IIP,
        S,
        T,
    }
    return nothing
end

@inline function DiffEqBase.get_condition(
        integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S, T,
        },
        callback,
        abst
    ) where {
        AlgType <: GPUODEAlgorithm, IIP, S, T,
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
