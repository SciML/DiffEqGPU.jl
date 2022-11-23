function vectorized_solve(probs, prob::ODEProblem, alg;
                          dt, saveat = nothing,
                          save_everystep = true,
                          debug = false, callback = nothing, tstops = nothing, kwargs...)
    # if saveat is specified, we'll use a vector of timestamps.
    # otherwise it's a matrix that may be different for each ODE.
    timeseries = prob.tspan[1]:dt:prob.tspan[2]
    nsteps = length(timeseries)
    if saveat === nothing
        if save_everystep
            len = length(prob.tspan[1]:dt:prob.tspan[2])
            if tstops !== nothing
                len += length(tstops) - count(x -> x in tstops, timeseries)
                nsteps += length(tstops) - count(x -> x in tstops, timeseries)
            end
        else
            len = 2
        end
        ts = CuMatrix{typeof(dt)}(undef, (len, length(probs)))
        us = CuMatrix{typeof(prob.u0)}(undef, (len, length(probs)))
    else
        saveat = CuArray{typeof(dt)}(saveat)
        ts = CuMatrix{typeof(dt)}(undef, (length(saveat), length(probs)))
        us = CuMatrix{typeof(prob.u0)}(undef, (length(saveat), length(probs)))
    end

    # Handle tstops
    tstops = cu(tstops)

    if alg isa GPUTsit5
        kernel = @cuda launch=false tsit5_kernel(probs, us, ts, dt, callback, tstops,
                                                 nsteps,
                                                 saveat, Val(save_everystep))
    elseif alg isa GPUVern7
        kernel = @cuda launch=false vern7_kernel(probs, us, ts, dt, callback, tstops,
                                                 nsteps,
                                                 saveat, Val(save_everystep))
    end
    if debug
        @show CUDA.registers(kernel)
        @show CUDA.memory(kernel)
    end

    config = launch_configuration(kernel.fun)
    threads = min(length(probs), config.threads)
    # XXX: this kernel performs much better with all blocks active
    blocks = max(cld(length(probs), threads), config.blocks)
    threads = cld(length(probs), blocks)

    kernel(probs, us, ts, dt, callback, tstops, nsteps, saveat; threads, blocks)

    # we build the actual solution object on the CPU because the GPU would create one
    # containig CuDeviceArrays, which we cannot use on the host (not GC tracked,
    # no useful operations, etc). That's unfortunate though, since this loop is
    # generally slower than the entire GPU execution, and necessitates synchronization
    #EDIT: Done when using with DiffEqGPU
    ts, us
end

function vectorized_asolve(probs, prob::ODEProblem, alg;
                           dt = 0.1f0, saveat = nothing,
                           save_everystep = false,
                           abstol = 1.0f-6, reltol = 1.0f-3,
                           debug = false, callback = nothing, tstops = nothing, kwargs...)
    # if saveat is specified, we'll use a vector of timestamps.
    # otherwise it's a matrix that may be different for each ODE.
    if saveat === nothing
        if save_everystep
            error("Don't use adaptive version with saveat == nothing and save_everystep = true")
        else
            len = 2
        end
        # if tstops !== nothing
        #     len += length(tstops)
        # end
        ts = CuMatrix{typeof(dt)}(undef, (len, length(probs)))
        us = CuMatrix{typeof(prob.u0)}(undef, (len, length(probs)))
    else
        saveat = CuArray{typeof(dt)}(saveat)
        ts = CuMatrix{typeof(dt)}(undef, (length(saveat), length(probs)))
        us = CuMatrix{typeof(prob.u0)}(undef, (length(saveat), length(probs)))
    end

    tstops = cu(tstops)

    if alg isa GPUTsit5
        kernel = @cuda launch=false atsit5_kernel(probs, us, ts, dt, callback, tstops,
                                                  abstol,
                                                  reltol,
                                                  saveat, Val(save_everystep))
    elseif alg isa GPUVern7
        kernel = @cuda launch=false avern7_kernel(probs, us, ts, dt, callback, tstops,
                                                  abstol,
                                                  reltol,
                                                  saveat, Val(save_everystep))
    end

    if debug
        @show CUDA.registers(kernel)
        @show CUDA.memory(kernel)
    end

    config = launch_configuration(kernel.fun)
    threads = min(length(probs), config.threads)
    # XXX: this kernel performs much better with all blocks active
    blocks = max(cld(length(probs), threads), config.blocks)
    threads = cld(length(probs), blocks)
    kernel(probs, us, ts, dt, callback, tstops, abstol, reltol, saveat; threads, blocks)

    # we build the actual solution object on the CPU because the GPU would create one
    # containig CuDeviceArrays, which we cannot use on the host (not GC tracked,
    # no useful operations, etc). That's unfortunate though, since this loop is
    # generally slower than the entire GPU execution, and necessitates synchronization
    #EDIT: Done when using with DiffEqGPU
    ts, us
end
