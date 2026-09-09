module CUDAExt

using CUDA: CUDA, CUDABackend
using KernelAbstractions: @kernel, @index
using Adapt: Adapt, adapt
using SciMLBase: SciMLBase
using DiffEqBase: DiffEqBase, CallbackSet, ReturnCode
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

# Enzyme needs numeric fields first and uninlined construction to retain aggregate type
# information. Defined only in CUDAExt so OpenCL/Metal never see an extra
# AbstractODEProblem subtype (Julia 1.12+ Dual SPIR-V / jl_genericmemory_copyto breaks).
struct KernelODEProblem{U, T, IIP, P, F, K, PT} <: SciMLBase.AbstractODEProblem{U, T, IIP}
    p::P
    u0::U
    tspan::T
    f::F
    kwargs::K
    problem_type::PT
end

_kernel_record(prob) = prob
@noinline function _kernel_record(
        prob::SciMLBase.ImmutableODEProblem{U, T, IIP, P, F, K, PT}
    ) where {U, T, IIP, P, F, K, PT}
    return KernelODEProblem{U, T, IIP, P, F, K, PT}(
        prob.p, prob.u0, prob.tspan, prob.f, prob.kwargs, prob.problem_type
    )
end

# EnzymeCUDAExt reverse of Ptr/CuPtr copies uses `.+=` on wrapped arrays; define fieldwise
# addition so KernelODEProblem shadows accumulate through host↔device copies.
function Base.:+(
        a::KernelODEProblem{U, T, IIP, P, F, K, PT},
        b::KernelODEProblem{U, T, IIP, P, F, K, PT}
    ) where {U, T, IIP, P, F, K, PT}
    return KernelODEProblem{U, T, IIP, P, F, K, PT}(
        a.p + b.p, a.u0 + b.u0, a.tspan, a.f, a.kwargs, a.problem_type
    )
end

function Adapt.adapt_structure(
        to, prob::KernelODEProblem{U, T, IIP, P, F, K, PT}
    ) where {U, T, IIP, P, F, K, PT}
    # Only adapt numeric payload. Adapting `f` can change ODEFunction specialize
    # (AutoSpecialize → FullSpecialize) and break the KernelODEProblem type params.
    p = adapt(to, prob.p)
    u0 = adapt(to, prob.u0)
    tspan = adapt(to, prob.tspan)
    return KernelODEProblem{typeof(u0), typeof(tspan), IIP, typeof(p), F, K, PT}(
        p, u0, tspan, prob.f, prob.kwargs, prob.problem_type
    )
end

@noinline function _make_kernel_problem(ensembleprob, i, sim_seeds, rng_func, master_rng)
    ctx = DiffEqGPU._make_ensemble_context(i, sim_seeds, rng_func, master_rng)
    prob = ensembleprob.safetycopy ? deepcopy(ensembleprob.prob) : ensembleprob.prob
    return DiffEqGPU.make_prob_compatible(ensembleprob.prob_func(prob, ctx))
end

function _prepare_kernel_problems(ensembleprob, backend, I, sim_seeds, rng_func, master_rng)
    first_prob = _make_kernel_problem(ensembleprob, first(I), sim_seeds, rng_func, master_rng)
    # Host Refs use KernelODEProblem so Enzyme can shadow build_solution's problem arg.
    # Device copy goes through ImmutableODEProblem adapt (preserves ODEFunction specialize)
    # then _kernel_record — do not adapt(KernelODEProblem) for the first device copy.
    first_host = _kernel_record(first_prob)
    first_adapted = _kernel_record(adapt(backend, first_prob))
    first_ref = Ref(first_host)
    probs = Vector{typeof(first_ref)}(undef, length(I))
    adapted_probs = Vector{typeof(first_adapted)}(undef, length(I))
    probs[1] = first_ref
    adapted_probs[1] = first_adapted
    # Adapt during construction: a separate broadcast over isbits problems loses Enzyme gradients.
    for j in 2:length(I)
        prob = _make_kernel_problem(ensembleprob, I[j], sim_seeds, rng_func, master_rng)
        host = _kernel_record(prob)
        probs[j] = Ref(host)
        adapted_probs[j] = _kernel_record(adapt(backend, prob))
    end
    return probs, adapted_probs
end

function _batch_solve_up_kernel_enzyme(
        ensembleprob, probs, adapted_probs, alg, ensemblealg, I, adaptive;
        kwargs...
    )
    _callback = CallbackSet(
        DiffEqGPU.generate_callback(probs[1][], length(I), ensemblealg; kwargs...)
    )

    _callback = CallbackSet(
        convert.(DiffEqGPU.GPUDiscreteCallback, _callback.discrete_callbacks)...,
        convert.(DiffEqGPU.GPUContinuousCallback, _callback.continuous_callbacks)...
    )

    dev = ensemblealg.dev
    probs = adapt(dev, adapted_probs)

    if adaptive
        ts, us = DiffEqGPU.vectorized_asolve(
            probs, ensembleprob.prob, alg;
            kwargs..., callback = _callback
        )
    else
        ts, us = DiffEqGPU.vectorized_solve(
            probs, ensembleprob.prob, alg;
            kwargs..., callback = _callback
        )
    end
    return Array(ts), Array(us)
end

function DiffEqGPU.batch_solve_gpukernel(
        ensembleprob,
        alg,
        ensemblealg::DiffEqGPU.EnsembleGPUKernel{<:CUDABackend},
        I,
        adaptive;
        sim_seeds = nothing,
        rng_func = SciMLBase.default_rng_func,
        master_rng = nothing,
        kwargs...
    )
    kernel_probs, adapted_kernel_probs = _prepare_kernel_problems(
        ensembleprob, ensemblealg.dev, I, sim_seeds, rng_func, master_rng
    )
    if !all(
            Base.Fix2(
                (prob1, prob2) -> isequal(prob1[].tspan, prob2[].tspan),
                kernel_probs[1]
            ),
            kernel_probs
        )
        if !iszero(ensemblealg.cpu_offload)
            error("Different time spans in an Ensemble Simulation with CPU offloading is not supported yet.")
        end
        if get(kernel_probs[1][].kwargs, :saveat, nothing) === nothing && !adaptive &&
                get(kwargs, :save_everystep, true)
            error("Using different time-spans require either turning off save_everystep or using saveat. If using saveat, it should be of same length across the ensemble.")
        end
        if !all(
                Base.Fix2(
                    (
                        prob1,
                        prob2,
                    ) -> isequal(
                        sizeof(get(prob1[].kwargs, :saveat, nothing)),
                        sizeof(get(prob2[].kwargs, :saveat, nothing))
                    ),
                    kernel_probs[1]
                ),
                kernel_probs
            )
            error("Using different saveat in EnsembleGPUKernel requires all of them to be of same length. Use saveats of same size only.")
        end
    end

    if !(alg isa Union{DiffEqGPU.GPUODEAlgorithm, DiffEqGPU.GPUSDEAlgorithm})
        error("We don't have solvers implemented for this algorithm yet")
    end

    _saveat = get(kernel_probs[1][].kwargs, :saveat, nothing)
    saveat = _saveat === nothing ? get(kwargs, :saveat, nothing) : _saveat
    solts, kernel_solus = _batch_solve_up_kernel_enzyme(
        ensembleprob, kernel_probs, adapted_kernel_probs, alg, ensemblealg, I,
        adaptive; saveat, kwargs...
    )
    return [
        begin
            ts = @view solts[:, i]
            us = @view kernel_solus[:, i]
            sol_idx = findlast(x -> x != kernel_probs[i][].tspan[1], ts)
            if sol_idx === nothing
                @error "No solution found" tspan = kernel_probs[i][].tspan[1] ts
                error("Batch solve failed")
            end
            @views ensembleprob.output_func(
                SciMLBase.build_solution(
                    kernel_probs[i][],
                    alg,
                    ts[1:sol_idx],
                    us[1:sol_idx],
                    k = nothing,
                    stats = nothing,
                    calculate_error = false,
                    retcode = sol_idx != length(ts) ? ReturnCode.Terminated :
                        ReturnCode.Success
                ),
                DiffEqGPU._make_ensemble_context(I[i], sim_seeds, rng_func, master_rng)
            )[1]
        end
            for i in eachindex(kernel_probs)
    ]
end

end
