function generate_callback(callback::ContinuousCallback, I, ensemblealg)
    if ensemblealg isa EnsembleGPUKernel
        return callback
    end
    _condition = callback.condition
    _affect! = callback.affect!
    _affect_neg! = callback.affect_neg!

    condition = function (out, u, t, integrator)
        version = get_backend(u)
        wgs = workgroupsize(version, size(u, 2))
        continuous_condition_kernel(version)(
            _condition, out, u, t, integrator.p;
            ndrange = size(u, 2),
            workgroupsize = wgs
        )
        return nothing
    end

    # DiffEqBase v7's `apply_callback!` for `VectorContinuousCallback` invokes
    # `callback.affect!(integrator, simultaneous_events::Vector{Int8})` once per
    # step. Each entry of the mask is 0 (no trigger), -1 (upcrossing) or
    # +1 (downcrossing) — see OrdinaryDiffEq v7 NEWS.md, "Breaking:
    # VectorContinuousCallback affect! signature changed". We copy the host
    # mask to a backend-native array, dispatch one GPU thread per trajectory,
    # and route up/down crossings to the user's original `affect!` /
    # `affect_neg!`. v7 no longer dispatches to `VectorContinuousCallback`'s
    # `affect_neg!` field, so we don't supply one.
    affect! = function (integrator, simultaneous_events::AbstractVector)
        version = get_backend(integrator.u)
        wgs = workgroupsize(version, size(integrator.u, 2))
        se_device = similar(
            integrator.u, eltype(simultaneous_events),
            length(simultaneous_events)
        )
        copyto!(se_device, simultaneous_events)
        return continuous_affect!_kernel(version)(
            _affect!, _affect_neg!, se_device, integrator.u,
            integrator.t, integrator.p;
            ndrange = size(integrator.u, 2),
            workgroupsize = wgs
        )
    end

    return VectorContinuousCallback(
        condition, affect!, I,
        save_positions = callback.save_positions
    )
end

function generate_callback(callback::CallbackSet, I, ensemblealg)
    return CallbackSet(
        map(
            cb -> generate_callback(cb, I, ensemblealg),
            (
                callback.continuous_callbacks...,
                callback.discrete_callbacks...,
            )
        )...
    )
end

generate_callback(::Tuple{}, I, ensemblealg) = nothing

function generate_callback(x)
    # will catch any VectorContinuousCallbacks
    error("Callback unsupported")
end

function generate_callback(prob, I, ensemblealg; kwargs...)
    prob_cb = get(prob.kwargs, :callback, ())
    kwarg_cb = get(kwargs, :merge_callbacks, false) ? get(kwargs, :callback, ()) : ()

    if (prob_cb === nothing || isempty(prob_cb)) &&
            (kwarg_cb === nothing || isempty(kwarg_cb))
        return nothing
    else
        return CallbackSet(
            generate_callback(prob_cb, I, ensemblealg),
            generate_callback(kwarg_cb, I, ensemblealg)
        )
    end
end
