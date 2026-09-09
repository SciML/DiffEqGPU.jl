module CUDAExt
using CUDA: CUDA, CUDABackend
using KernelAbstractions: @kernel, @index
using DiffEqBase: DiffEqBase
using SciMLBase: SciMLBase
using SimpleNonlinearSolve: SimpleTrustRegion
import DiffEqGPU

function DiffEqGPU.EnsembleGPUArray(cpu_offload::Float64)
    return DiffEqGPU.EnsembleGPUArray(CUDABackend(), cpu_offload)
end
DiffEqGPU.maxthreads(::CUDABackend) = 256
DiffEqGPU.maybe_prefer_blocks(::CUDABackend) = CUDABackend(; prefer_blocks = true)

function DiffEqGPU.lufact!(::CUDABackend, W)
    CUDA.CUBLAS.getrf_strided_batched!(W, false)
    return nothing
end

# Enzyme AD kernels stay in this extension so OpenCL/Metal never see them.
@inline _load_kernel_arg(x::Number) = x
@inline _load_kernel_arg(x::AbstractArray{<:Any, 0}) = @inbounds x[]
@inline _load_kernel_arg(x::AbstractArray) = x

@kernel function ode_solve_kernel_ad(
        probs, alg, _us, _ts, _dt, callback,
        tstops,
        saveat, ::Val{save_everystep}
    ) where {save_everystep}
    i = @index(Global, Linear)
    dt = _load_kernel_arg(_dt)

    prob = @inbounds probs[i]
    ts = @inbounds view(_ts, :, i)
    us = @inbounds view(_us, :, i)

    _saveat = get(prob.kwargs, :saveat, nothing)
    saveat = _saveat === nothing ? saveat : _saveat

    u0, p_init,
        init_success = if SciMLBase.has_initialization_data(prob.f)
        DiffEqGPU.gpu_initialization_solve(prob, SimpleTrustRegion(), 1.0e-6, 1.0e-6)
    else
        prob.u0, prob.p, true
    end

    if init_success
        integ = DiffEqGPU.init(
            alg, prob.f, false, u0, prob.tspan[1], dt, p_init, tstops,
            callback, save_everystep, saveat
        )
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
            @inbounds us[integ.step_idx] = u0
        end

        integ.step_idx += 1
        while integ.t < tspan[2] && integ.retcode != DiffEqBase.ReturnCode.Terminated
            saved_in_cb = DiffEqGPU.step!(integ, ts, us)
            !saved_in_cb && DiffEqGPU.savevalues!(integ, ts, us)
        end
        if saveat === nothing && !save_everystep
            @inbounds us[2] = integ.u
            @inbounds ts[2] = integ.t
        end
        if integ.t > tspan[2] && saveat === nothing
            @inbounds us[end] = integ(tspan[2])
            @inbounds ts[end] = tspan[2]
        end
    else
        @inbounds us[1] = prob.u0
        @inbounds ts[1] = prob.tspan[1]
        if saveat === nothing && !save_everystep
            @inbounds us[2] = prob.u0
            @inbounds ts[2] = prob.tspan[1]
        end
    end
end

@kernel function ode_asolve_kernel_ad(
        probs, alg, _us, _ts, _dt, callback, tstops,
        _abstol, _reltol,
        saveat,
        ::Val{save_everystep}
    ) where {save_everystep}
    i = @index(Global, Linear)
    dt = _load_kernel_arg(_dt)
    abstol = _load_kernel_arg(_abstol)
    reltol = _load_kernel_arg(_reltol)

    prob = @inbounds probs[i]
    ts = @inbounds view(_ts, :, i)
    us = @inbounds view(_us, :, i)

    _saveat = get(prob.kwargs, :saveat, nothing)
    saveat = _saveat === nothing ? saveat : _saveat

    u0, p_init,
        init_success = if SciMLBase.has_initialization_data(prob.f)
        DiffEqGPU.gpu_initialization_solve(prob, SimpleTrustRegion(), abstol, reltol)
    else
        prob.u0, prob.p, true
    end

    if init_success
        tspan = prob.tspan
        integ = DiffEqGPU.init(
            alg, prob.f, false, u0, prob.tspan[1], prob.tspan[2], dt,
            p_init,
            abstol, reltol, DiffEqBase.ODE_DEFAULT_NORM, tstops, callback,
            saveat
        )

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
            saved_in_cb = DiffEqGPU.step!(integ, ts, us)
            !saved_in_cb && DiffEqGPU.savevalues!(integ, ts, us)
        end

        if integ.t > tspan[2] && saveat === nothing
            @inbounds us[end] = integ(tspan[2])
            @inbounds ts[end] = tspan[2]
        end

        if saveat === nothing && !save_everystep
            @inbounds us[2] = integ.u
            @inbounds ts[2] = integ.t
        end
    else
        @inbounds us[1] = prob.u0
        @inbounds ts[1] = prob.tspan[1]
        if saveat === nothing && !save_everystep
            @inbounds us[2] = prob.u0
            @inbounds ts[2] = prob.tspan[1]
        end
    end
end

function DiffEqGPU._cuda_ode_solve_kernel_ad(backend::CUDABackend)
    return ode_solve_kernel_ad(backend)
end

function DiffEqGPU._cuda_ode_asolve_kernel_ad(backend::CUDABackend)
    return ode_asolve_kernel_ad(backend)
end

end
