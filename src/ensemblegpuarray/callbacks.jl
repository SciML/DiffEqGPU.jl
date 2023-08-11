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
        continuous_condition_kernel(version)(_condition, out, u, t, integrator.p;
            ndrange = size(u, 2),
            workgroupsize = wgs)
        nothing
    end

    affect! = function (integrator, event_idx)
        version = get_backend(integrator.u)
        wgs = workgroupsize(version, size(integrator.u, 2))
        continuous_affect!_kernel(version)(_affect!, event_idx, integrator.u,
            integrator.t, integrator.p;
            ndrange = size(integrator.u, 2),
            workgroupsize = wgs)
    end

    affect_neg! = function (integrator, event_idx)
        version = get_backend(integrator.u)
        wgs = workgroupsize(version, size(integrator.u, 2))
        continuous_affect!_kernel(version)(_affect_neg!, event_idx, integrator.u,
            integrator.t, integrator.p;
            ndrange = size(integrator.u, 2),
            workgroupsize = wgs)
    end

    return VectorContinuousCallback(condition, affect!, affect_neg!, I,
        save_positions = callback.save_positions)
end

function generate_callback(callback::CallbackSet, I, ensemblealg)
    return CallbackSet(map(cb -> generate_callback(cb, I, ensemblealg),
        (callback.continuous_callbacks...,
            callback.discrete_callbacks...))...)
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
        return CallbackSet(generate_callback(prob_cb, I, ensemblealg),
            generate_callback(kwarg_cb, I, ensemblealg))
    end
end