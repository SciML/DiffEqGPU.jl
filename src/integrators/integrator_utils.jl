function build_adaptive_tsit5_controller_cache(::Type{T}) where {T}
    beta1 = T(7 / 50)
    beta2 = T(2 / 25)
    qmax = T(10.0)
    qmin = T(1 / 5)
    gamma = T(9 / 10)
    qoldinit = T(1e-4)
    qold = qoldinit

    return beta1, beta2, qmax, qmin, gamma, qoldinit, qold
end

function savevalues!(integrator::GPUTsit5Integrator, ts, us, force = false)
    saved, savedexactly = false, false

    if integrator.save_everystep || force
        saved = true
        savedexactly = true
        @inbounds us[integrator.step_idx] = integrator.u
        @inbounds ts[integrator.step_idx] = integrator.t
        integrator.step_idx += 1
    end

    saved, savedexactly
end

@inline function DiffEqBase.apply_discrete_callback!(integrator::GPUATsit5Integrator,
                                                     callback::GPUDiscreteCallback)
    saved_in_cb = false
    if callback.condition(integrator.u, integrator.t, integrator)
        integrator.u_modified = true
        callback.affect!(integrator)
    end
    integrator.u_modified, saved_in_cb
end

#Starting: Get bool from first and do next
@inline function DiffEqBase.apply_discrete_callback!(integrator,
                                                     callback::GPUDiscreteCallback, args...)
    DiffEqBase.apply_discrete_callback!(integrator,
                                        DiffEqBase.apply_discrete_callback!(integrator,
                                                                            callback)...,
                                        args...)
end

@inline function DiffEqBase.apply_discrete_callback!(integrator, discrete_modified::Bool,
                                                     saved_in_cb::Bool,
                                                     callback::GPUDiscreteCallback,
                                                     args...)
    bool, saved_in_cb2 = DiffEqBase.apply_discrete_callback!(integrator,
                                                             DiffEqBase.apply_discrete_callback!(integrator,
                                                                                                 callback)...,
                                                             args...)
    discrete_modified || bool, saved_in_cb || saved_in_cb2
end

@inline function DiffEqBase.apply_discrete_callback!(integrator, discrete_modified::Bool,
                                                     saved_in_cb::Bool,
                                                     callback::GPUDiscreteCallback)
    bool, saved_in_cb2 = DiffEqBase.apply_discrete_callback!(integrator, callback)
    discrete_modified || bool, saved_in_cb || saved_in_cb2
end

@inline function apply_discrete_callback!(integrator::GPUTsit5Integrator, ts, us,
                                          callback::GPUDiscreteCallback)
    saved_in_cb = false
    if callback.condition(integrator.u, integrator.t, integrator)
        # handle saveat
        _, savedexactly = savevalues!(integrator, ts, us)
        saved_in_cb = true
        @inbounds if callback.save_positions[1]
            # if already saved then skip saving
            savedexactly || savevalues!(integrator, ts, us, true)
        end
        integrator.u_modified = true
        callback.affect!(integrator)
        @inbounds if callback.save_positions[2]
            savevalues!(integrator, ts, us, true)
            saved_in_cb = true
        end
    end
    integrator.u_modified, saved_in_cb
end

@inline function apply_discrete_callback!(integrator::GPUTsit5Integrator, ts, us,
                                          callback::GPUDiscreteCallback,
                                          args...)
    apply_discrete_callback!(integrator, ts, us,
                             apply_discrete_callback!(integrator, ts, us, callback)...,
                             args...)
end

@inline function apply_discrete_callback!(integrator::GPUTsit5Integrator, ts, us,
                                          discrete_modified::Bool,
                                          saved_in_cb::Bool, callback::GPUDiscreteCallback,
                                          args...)
    bool, saved_in_cb2 = apply_discrete_callback!(integrator, ts, us,
                                                  apply_discrete_callback!(integrator, ts,
                                                                           us, callback)...,
                                                  args...)
    discrete_modified || bool, saved_in_cb || saved_in_cb2
end

@inline function apply_discrete_callback!(integrator::GPUTsit5Integrator, ts, us,
                                          discrete_modified::Bool,
                                          saved_in_cb::Bool, callback::GPUDiscreteCallback)
    bool, saved_in_cb2 = apply_discrete_callback!(integrator, ts, us, callback)
    discrete_modified || bool, saved_in_cb || saved_in_cb2
end
