"""
$(DocStringExtensions.README)
"""
module DiffEqGPU

using DocStringExtensions
using KernelAbstractions, CUDA, SciMLBase, DiffEqBase, LinearAlgebra, Distributed
using CUDAKernels
using CUDA: CuPtr, CU_NULL, Mem, CuDefaultStream
using CUDA: CUBLAS
using ForwardDiff
import ChainRulesCore
import ChainRulesCore: NoTangent
using RecursiveArrayTools
import ZygoteRules
import Base.Threads

@kernel function gpu_kernel(f,du,@Const(u),@Const(p),@Const(t))
    i = @index(Global, Linear)
    @views @inbounds f(du[:,i],u[:,i],p[:,i],t)
end

@kernel function gpu_kernel_oop(f,du,@Const(u),@Const(p),@Const(t))
    i = @index(Global, Linear)
    @views @inbounds x = f(u[:,i],p[:,i],t)
    @inbounds for j in 1:size(du,1)
        du[j,i] = x[j]
    end
end

@kernel function jac_kernel(f,J,@Const(u),@Const(p),@Const(t))
    i = @index(Global, Linear)-1
    section = 1 + (i*size(u,1)) : ((i+1)*size(u,1))
    @views @inbounds f(J[section,section],u[:,i+1],p[:,i+1],t)
end

@kernel function jac_kernel_oop(f,J,@Const(u),@Const(p),@Const(t))
    i = @index(Global, Linear)-1
    section = 1 + (i*size(u,1)) : ((i+1)*size(u,1))
    @views @inbounds x = f(u[:,i+1],p[:,i+1],t)
    @inbounds for j in section, k in section
        J[k,j] = x[k,j]
    end
end

@kernel function discrete_condition_kernel(condition,cur,@Const(u),@Const(t),@Const(p))
    i = @index(Global, Linear)
    @views @inbounds cur[i] = condition(u[:,i],t,FakeIntegrator(u[:,i],t,p[:,i]))
end

@kernel function discrete_affect!_kernel(affect!,cur,u,t,p)
    i = @index(Global, Linear)
    @views @inbounds cur[i] && affect!(FakeIntegrator(u[:,i],t,p[:,i]))
end

@kernel function continuous_condition_kernel(condition,out,@Const(u),@Const(t),@Const(p))
    i = @index(Global, Linear)
    @views @inbounds out[i] = condition(u[:,i],t,FakeIntegrator(u[:,i],t,p[:,i]))
end

@kernel function continuous_affect!_kernel(affect!,event_idx,u,t,p)
    for i in event_idx
        @views @inbounds affect!(FakeIntegrator(u[:,i],t,p[:,i]))
    end
end

maxthreads(::CPU) = 1024
maxthreads(::CUDADevice) = 256

function workgroupsize(backend, n)
    min(maxthreads(backend),n)
end

@kernel function W_kernel(jac, W, @Const(u), @Const(p), @Const(gamma), @Const(t))
    i = @index(Global, Linear)
    len = size(u,1)
    _W = @inbounds @view(W[:, :, i])
    @views @inbounds jac(_W,u[:,i],p[:,i],t)
    @inbounds for i in eachindex(_W)
        _W[i] = gamma*_W[i]
    end
    _one = one(eltype(_W))
    @inbounds for i in 1:len
        _W[i, i] = _W[i, i] - _one
    end
end

@kernel function W_kernel_oop(jac, W, @Const(u), @Const(p), @Const(gamma), @Const(t))
    i = @index(Global, Linear)
    len = size(u,1)
    _W = @inbounds @view(W[:, :, i])
    @views @inbounds x = jac(u[:,i],p[:,i],t)
    @inbounds for j in 1:length(_W)
        _W[j] = x[j]
    end
    @inbounds for i in eachindex(_W)
        _W[i] = gamma*_W[i]
    end
    _one = one(eltype(_W))
    @inbounds for i in 1:len
        _W[i, i] = _W[i, i] - _one
    end
end

@kernel function Wt_kernel(jac, W, @Const(u), @Const(p), @Const(gamma), @Const(t))
    i = @index(Global, Linear)
    len = size(u,1)
    _W = @inbounds @view(W[:, :, i])
    @views @inbounds jac(_W,u[:,i],p[:,i],t)
    @inbounds for i in 1:len
        _W[i, i] = -inv(gamma) + _W[i, i]
    end
end

@kernel function Wt_kernel_oop(jac, W, @Const(u), @Const(p), @Const(gamma), @Const(t))
    i = @index(Global, Linear)
    len = size(u,1)
    _W = @inbounds @view(W[:, :, i])
    @views @inbounds x = jac(u[:,i],p[:,i],t)
    @inbounds for j in 1:length(_W)
        _W[j] = x[j]
    end
    @inbounds for i in 1:len
        _W[i, i] = -inv(gamma) + _W[i, i]
    end
end

function cuda_lufact!(W)
    CUBLAS.getrf_strided_batched!(W, false)
    return nothing
end

function cpu_lufact!(W)
    len = size(W, 1)
    for i in 1:size(W,3)
        _W = @inbounds @view(W[:, :, i])
        generic_lufact!(_W, len)
    end
    return nothing
end

struct FakeIntegrator{uType,tType,P}
    u::uType
    t::tType
    p::P
end

abstract type EnsembleArrayAlgorithm <: SciMLBase.EnsembleAlgorithm end
struct EnsembleCPUArray <: EnsembleArrayAlgorithm end
struct EnsembleGPUArray <: EnsembleArrayAlgorithm
    cpu_offload::Float64
end

# Work around the fact that Zygote cannot handle the task system
# Work around the fact that Zygote isderiving fails with constants?
function EnsembleGPUArray()
    EnsembleGPUArray(0.2)
end

function ChainRulesCore.rrule(::Type{<:EnsembleGPUArray})
    EnsembleGPUArray(0.0), _ -> NoTangent()
end

ZygoteRules.@adjoint function EnsembleGPUArray()
    EnsembleGPUArray(0.0), _ -> nothing
end

function SciMLBase.__solve(ensembleprob::SciMLBase.AbstractEnsembleProblem,
                 alg::Union{SciMLBase.DEAlgorithm,Nothing},
                 ensemblealg::EnsembleArrayAlgorithm;
                 trajectories, batch_size = trajectories,
                 unstable_check = (dt,u,p,t)->false,
                 kwargs...)

    if trajectories == 1
        return SciMLBase.__solve(ensembleprob,alg,EnsembleSerial();trajectories=1,kwargs...)
    end

    cpu_trajectories = (ensemblealg isa EnsembleGPUArray && ensembleprob.reduction === SciMLBase.DEFAULT_REDUCTION) ? round(Int,trajectories * ensemblealg.cpu_offload) : 0
    gpu_trajectories = trajectories - cpu_trajectories

    num_batches = gpu_trajectories ÷ batch_size
    num_batches * batch_size != gpu_trajectories && (num_batches += 1)

    if cpu_trajectories != 0 && ensembleprob.reduction === SciMLBase.DEFAULT_REDUCTION

        cpu_II = (gpu_trajectories+1):trajectories
        function f()
            SciMLBase.solve_batch(ensembleprob,alg,EnsembleThreads(),cpu_II,nothing;kwargs...)
        end

        cpu_sols = Channel{Core.Compiler.return_type(f,Tuple{})}(1)
        t = @task begin
            put!(cpu_sols,f())
        end
        schedule(t)
    end


    if num_batches == 1 && ensembleprob.reduction === SciMLBase.DEFAULT_REDUCTION
       time = @elapsed sol = batch_solve(ensembleprob,alg,ensemblealg,1:gpu_trajectories;unstable_check=unstable_check,kwargs...)
       if cpu_trajectories != 0
         wait(t)
         sol = vcat(sol,take!(cpu_sols))
       end
       return SciMLBase.EnsembleSolution(sol,time,true)
    end

    converged::Bool = false
    u = ensembleprob.u_init === nothing ? similar(batch_solve(ensembleprob,alg,ensemblealg,1:batch_size;unstable_check=unstable_check,kwargs...), 0) : ensembleprob.u_init

    if nprocs() == 1
        # While pmap works, this makes much better error messages.
        time = @elapsed begin
            sols = map(1:num_batches) do i
                if i == num_batches
                  I = (batch_size*(i-1)+1):gpu_trajectories
                else
                  I = (batch_size*(i-1)+1):batch_size*i
                end
                batch_data = batch_solve(ensembleprob,alg,ensemblealg,I;unstable_check=unstable_check,kwargs...)
                if ensembleprob.reduction !== SciMLBase.DEFAULT_REDUCTION
                  u, _ = ensembleprob.reduction(u,batch_data,I)
                  return u
                else
                  batch_data
                end
            end
        end
    else
        time = @elapsed begin
            sols = pmap(1:num_batches) do i
                if i == num_batches
                  I = (batch_size*(i-1)+1):gpu_trajectories
                else
                  I = (batch_size*(i-1)+1):batch_size*i
                end
                x = batch_solve(ensembleprob,alg,ensemblealg,I;unstable_check=unstable_check,kwargs...)
                yield()
                if ensembleprob.reduction !== SciMLBase.DEFAULT_REDUCTION
                  u, _ = ensembleprob.reduction(u,x,I)
                else
                  x
                end

            end
        end
    end

    if ensembleprob.reduction === SciMLBase.DEFAULT_REDUCTION
        if cpu_trajectories != 0
          wait(t)
          sols = vcat(reduce(vcat,vec.(sols)),take!(cpu_sols))
        else
          sols = reduce(vcat,sols)
        end
        SciMLBase.EnsembleSolution(sols,time,true)
    else
        SciMLBase.EnsembleSolution(sols[end], time, true)
    end
end

diffeqgpunorm(u::AbstractArray,t) = sqrt.(sum(abs2, u)./length(u))
diffeqgpunorm(u::Union{AbstractFloat,Complex},t) = abs(u)
diffeqgpunorm(u::AbstractArray{<:ForwardDiff.Dual},t) = sqrt.(sum(abs2∘ForwardDiff.value, u)./length(u))
diffeqgpunorm(u::ForwardDiff.Dual,t) = abs(ForwardDiff.value(u))

function batch_solve(ensembleprob,alg,ensemblealg::EnsembleArrayAlgorithm,I;kwargs...)
    safetycopy = (ensembleprob.safetycopy === nothing) : SciMLBase.DEFAULT_SAFETYCOPY(ensembleprob.prob.prob_func) : ensembleprob.safetycopy
    if safetycopy
        probs = map(I) do i
            ensembleprob.prob_func(deepcopy(ensembleprob.prob),i,1)
        end
    else
        probs = map(I) do i
            ensembleprob.prob_func(ensembleprob.prob,i,1)
        end
    end
    @assert all(p->p.tspan == probs[1].tspan,probs)
    @assert !isempty(I)
    #@assert all(p->p.f === probs[1].f,probs)

    u0 = reduce(hcat,Array(probs[i].u0) for i in 1:length(I))
    p  = reduce(hcat,probs[i].p isa SciMLBase.NullParameters ? probs[i].p : Array(probs[i].p)  for i in 1:length(I))
    sol, solus = batch_solve_up(ensembleprob,probs,alg,ensemblealg,I,u0,p;kwargs...)
    [ensembleprob.output_func(SciMLBase.build_solution(probs[i],alg,sol.t,solus[i],destats=sol.destats,retcode=sol.retcode),i)[1] for i in 1:length(probs)]
end

function batch_solve_up(ensembleprob,probs,alg,ensemblealg,I,u0,p;kwargs...)
    if ensemblealg isa EnsembleGPUArray
        u0 = CuArray(u0)
        p  = CuArray(p)
    end

    len = length(probs[1].u0)

    if SciMLBase.has_jac(probs[1].f)
        jac_prototype = ensemblealg isa EnsembleGPUArray ?
                        cu(zeros(Float32,len,len,length(I))) :
                        zeros(Float32,len,len,length(I))
        if probs[1].f.colorvec !== nothing
            colorvec = repeat(probs[1].f.colorvec,length(I))
        else
            colorvec = repeat(1:length(probs[1].u0),length(I))
        end
    else
        jac_prototype = nothing
        colorvec = nothing
    end

    _callback = generate_callback(probs[1],length(I),ensemblealg; kwargs...)
    prob = generate_problem(probs[1],u0,p,jac_prototype,colorvec)

    if hasproperty(alg, :linsolve)
        _alg = remake(alg,linsolve = LinSolveGPUSplitFactorize())
    else
        _alg = alg
    end

    sol  = solve(prob,_alg; kwargs..., callback = _callback,merge_callbacks = false,
                 internalnorm=diffeqgpunorm)

    us = Array.(sol.u)
    solus = [[@view(us[i][:,j]) for i in 1:length(us)] for j in 1:length(probs)]
    (sol,solus)
end

function seed_duals(x::Matrix{V},::Type{T},
                    ::ForwardDiff.Chunk{N} = ForwardDiff.Chunk(@view(x[:,1]),typemax(Int64)),
                    ) where {V,T,N}
  seeds = ForwardDiff.construct_seeds(ForwardDiff.Partials{N,V})
  duals = [ForwardDiff.Dual{T}(x[i,j],seeds[i]) for i in 1:size(x,1), j in 1:size(x,2)]
end

function extract_dus(us)
  jsize = size(us[1],1), ForwardDiff.npartials(us[1][1])
  utype = typeof(ForwardDiff.value(us[1][1]))
  map(1:size(us[1],2)) do k
      map(us) do u
        du_i = zeros(utype, jsize)
        for i in size(u,1)
          du_i[i, :] = ForwardDiff.partials(u[i,k])
        end
        du_i
      end
  end
end

struct DiffEqGPUAdjTag end

function ChainRulesCore.rrule(::typeof(batch_solve_up),ensembleprob,probs,alg,ensemblealg,I,u0,p;kwargs...)
    pdual = seed_duals(p,DiffEqGPUAdjTag)
    u0 = convert.(eltype(pdual),u0)

    if ensemblealg isa EnsembleGPUArray
        u0     = CuArray(u0)
        pdual  = CuArray(pdual)
    end

    len = length(probs[1].u0)

    if SciMLBase.has_jac(probs[1].f)
        jac_prototype = ensemblealg isa EnsembleGPUArray ?
                        cu(zeros(Float32,len,len,length(I))) :
                        zeros(Float32,len,len,length(I))
        if probs[1].f.colorvec !== nothing
            colorvec = repeat(probs[1].f.colorvec,length(I))
        else
            colorvec = repeat(1:length(probs[1].u0),length(I))
        end
    else
        jac_prototype = nothing
        colorvec = nothing
    end

    _callback = generate_callback(probs[1],length(I),ensemblealg)
    prob = generate_problem(probs[1],u0,pdual,jac_prototype,colorvec)

    if hasproperty(alg, :linsolve)
        _alg = remake(alg,linsolve = LinSolveGPUSplitFactorize())
    else
        _alg = alg
    end

    sol  = solve(prob,_alg; kwargs..., callback = _callback,merge_callbacks = false,
                 internalnorm=diffeqgpunorm)

    us = Array.(sol.u)
    solus = [[ForwardDiff.value.(@view(us[i][:,j])) for i in 1:length(us)] for j in 1:length(probs)]

    function batch_solve_up_adjoint(Δ)
        dus = extract_dus(us)
        _Δ = Δ[2]
        adj = map(eachindex(dus)) do j
            sum(eachindex(dus[j])) do i
                J = dus[j][i]
                if _Δ[j] isa AbstractVector
                    v = _Δ[j][i]
                else
                    v = @view _Δ[j][i]
                end
                J'v
            end
        end
        (ntuple(_->NoTangent(), 7)...,Array(VectorOfArray(adj)))
    end
    (sol,solus),batch_solve_up_adjoint
end

function generate_problem(prob::ODEProblem,u0,p,jac_prototype,colorvec)
    _f = let f=prob.f.f, kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop
        function (du,u,p,t)
            version = u isa CuArray ? CUDADevice() : CPU()
            wgs = workgroupsize(version,size(u,2))
            wait(version, kernel(version)(f,du,u,p,t;ndrange=size(u,2),
                                              dependencies=Event(version),
                                              workgroupsize=wgs))
        end
    end

    if SciMLBase.has_jac(prob.f)
        _Wfact! = let jac=prob.f.jac, kernel = DiffEqBase.isinplace(prob) ? W_kernel : W_kernel_oop
            function (W,u,p,gamma,t)
                iscuda = u isa CuArray
                version = iscuda ? CUDADevice() : CPU()
                wgs = workgroupsize(version,size(u,2))
                wait(version, kernel(version)(jac, W, u, p, gamma, t;
                                                ndrange=size(u,2),
                                                dependencies=Event(version),
                                                workgroupsize=wgs))
                iscuda ? cuda_lufact!(W) : cpu_lufact!(W)
            end
        end
        _Wfact!_t = let jac=prob.f.jac, kernel = DiffEqBase.isinplace(prob) ? Wt_kernel : Wt_kernel_oop
            function (W,u,p,gamma,t)
                iscuda = u isa CuArray
                version = iscuda ? CUDADevice() : CPU()
                wgs = workgroupsize(version,size(u,2))
                wait(version, kernel(version)(jac, W, u, p, gamma, t;
                                                 ndrange=size(u,2),
                                                 dependencies=Event(version),
                                                 workgroupsize=wgs))
                iscuda ? cuda_lufact!(W) : cpu_lufact!(W)
            end
        end
    else
        _Wfact! = nothing
        _Wfact!_t = nothing
    end

    if SciMLBase.has_tgrad(prob.f)
        _tgrad = let tgrad=prob.f.tgrad, kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop
            function (J,u,p,t)
                version = u isa CuArray ? CUDADevice() : CPU()
                wgs = workgroupsize(version,size(u,2))
                wait(version, kernel(version)(tgrad,J,u,p,t;
                                                  ndrange=size(u,2),
                                                  dependencies=Event(version),
                                                  workgroupsize=wgs))
            end
        end
    else
        _tgrad = nothing
    end

    f_func = ODEFunction(_f,Wfact = _Wfact!,
                        Wfact_t = _Wfact!_t,
                        #colorvec=colorvec,
                        jac_prototype = jac_prototype,
                        tgrad=_tgrad)
    prob = ODEProblem(f_func,u0,prob.tspan,p;
                      prob.kwargs...)
end

function generate_problem(prob::SDEProblem,u0,p,jac_prototype,colorvec)
    _f = let f=prob.f.f, kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop
        function (du,u,p,t)
            version = u isa CuArray ? CUDADevice() : CPU()
            wgs = workgroupsize(version,size(u,2))
            wait(version, kernel(version)(f,du,u,p,t;
                                              ndrange=size(u,2),
                                              dependencies=Event(version),
                                              workgroupsize=wgs))
        end
    end

    _g = let f=prob.f.g, kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop
        function (du,u,p,t)
            version = u isa CuArray ? CUDADevice() : CPU()
            wgs = workgroupsize(version,size(u,2))
            wait(version, kernel(version)(f,du,u,p,t;
                                              ndrange=size(u,2),
                                              dependencies=Event(version),
                                              workgroupsize=wgs))
        end
    end

    if SciMLBase.has_jac(prob.f)
        _Wfact! = let jac=prob.f.jac, kernel = DiffEqBase.isinplace(prob) ? W_kernel : W_kernel_oop
            function (W,u,p,gamma,t)
                iscuda = u isa CuArray
                version = iscuda ? CUDADevice() : CPU()
                wgs = workgroupsize(version,size(u,2))
                wait(version, kernel(version)(jac, W, u, p, gamma, t;
                                                ndrange=size(u,2),
                                                dependencies=Event(version),
                                                workgroupsize=wgs))
                iscuda ? cuda_lufact!(W) : cpu_lufact!(W)
            end
        end
        _Wfact!_t = let jac=prob.f.jac, kernel = DiffEqBase.isinplace(prob) ? Wt_kernel : Wt_kernel_oop
            function (W,u,p,gamma,t)
                iscuda = u isa CuArray
                version = iscuda ? CUDADevice() : CPU()
                wgs = workgroupsize(version,size(u,2))
                wait(version, kernel(version)(jac, W, u, p, gamma, t;
                                                 ndrange=size(u,2),
                                                 dependencies=Event(version),
                                                 workgroupsize=wgs))
                iscuda ? cuda_lufact!(W) : cpu_lufact!(W)
            end
        end
    else
        _Wfact! = nothing
        _Wfact!_t = nothing
    end

    if SciMLBase.has_tgrad(prob.f)
        _tgrad = let tgrad=prob.f.tgrad, kernel = DiffEqBase.isinplace(prob) ? gpu_kernel : gpu_kernel_oop
            function (J,u,p,t)
                version = u isa CuArray ? CUDADevice() : CPU()
                wgs = workgroupsize(version,size(u,2))
                wait(version, kernel(version)(tgrad,J,u,p,t;
                                                  ndrange=size(u,2),
                                                  dependencies=Event(version),
                                                  workgroupsize=wgs))
            end
        end
    else
        _tgrad = nothing
    end

    f_func = SDEFunction(_f,_g,Wfact = _Wfact!,
                        Wfact_t = _Wfact!_t,
                        #colorvec=colorvec,
                        jac_prototype = jac_prototype,
                        tgrad=_tgrad)
    prob = SDEProblem(f_func,_g,u0,prob.tspan,p;
                      prob.kwargs...)
end

function generate_callback(callback::DiscreteCallback,I,ensemblealg)
    if ensemblealg isa EnsembleGPUArray
        cur = CuArray([false for i in 1:I])
    else
        cur = [false for i in 1:I]
    end
    _condition = callback.condition
    _affect!   = callback.affect!

    condition = function (u,t,integrator)
        version = u isa CuArray ? CUDADevice() : CPU()
        wgs = workgroupsize(version,size(u,2))
        wait(version, discrete_condition_kernel(version)(_condition,cur,u,t,integrator.p;
                                                         ndrange=size(u,2),
                                                         dependencies=Event(version),
                                                         workgroupsize=wgs))
        any(cur)
    end

    affect! = function (integrator)
        version = integrator.u isa CuArray ? CUDADevice() : CPU()
        wgs = workgroupsize(version,size(integrator.u,2))
        wait(version, discrete_affect!_kernel(version)(_affect!,cur,integrator.u,integrator.t,integrator.p;
                                                       ndrange=size(integrator.u,2),
                                                       dependencies=Event(version),
                                                       workgroupsize=wgs))
    end

    return DiscreteCallback(condition,affect!,save_positions=callback.save_positions)
end

function generate_callback(callback::ContinuousCallback,I,ensemblealg)
    _condition   = callback.condition
    _affect!     = callback.affect!
    _affect_neg! = callback.affect_neg!

    condition = function (out,u,t,integrator)
        version = u isa CuArray ? CUDADevice() : CPU()
        wgs = workgroupsize(version,size(u,2))
        wait(version, continuous_condition_kernel(version)(_condition,out,u,t,integrator.p;
                                                           ndrange=size(u,2),
                                                           dependencies=Event(version),
                                                           workgroupsize=wgs))
        nothing
    end

    affect! = function (integrator,event_idx)
        version = integrator.u isa CuArray ? CUDADevice() : CPU()
        wgs = workgroupsize(version,size(integrator.u,2))
        wait(version, continuous_affect!_kernel(version)(_affect!,event_idx,integrator.u,integrator.t,integrator.p;
                                                         ndrange=size(integrator.u,2),
                                                         dependencies=Event(version),
                                                         workgroupsize=wgs))
    end

    affect_neg! = function (integrator,event_idx)
        version = integrator.u isa CuArray ? CUDADevice() : CPU()
        wgs = workgroupsize(version,size(integrator.u,2))
        wait(version, continuous_affect!_kernel(version)(_affect_neg!,event_idx,integrator.u,integrator.t,integrator.p;
                                                         ndrange=size(integrator.u,2),
                                                         dependencies=Event(version),
                                                         workgroupsize=wgs))
    end

    return VectorContinuousCallback(condition,affect!,affect_neg!,I,save_positions=callback.save_positions)
end

function generate_callback(callback::CallbackSet,I,ensemblealg)
    return CallbackSet(map(cb->generate_callback(cb,I,ensemblealg),
            (callback.continuous_callbacks..., callback.discrete_callbacks...))...)
end

generate_callback(::Tuple{},I,ensemblealg) = nothing

function generate_callback(x)
    # will catch any VectorContinuousCallbacks
    error("Callback unsupported")
end

function generate_callback(prob,I,ensemblealg; kwargs...)
    prob_cb = get(prob.kwargs, :callback, ())
    kwarg_cb = get(kwargs, :merge_callbacks, false) ? get(kwargs, :callback, ()) : ()

    if isempty(prob_cb) && isempty(kwarg_cb)
        return nothing
    else
        return CallbackSet(generate_callback(prob_cb,I,ensemblealg),
                           generate_callback(kwarg_cb,I,ensemblealg))
    end
end

### GPU Factorization

struct LinSolveGPUSplitFactorize
    len::Int
    nfacts::Int
end
LinSolveGPUSplitFactorize() = LinSolveGPUSplitFactorize(0, 0)

function (p::LinSolveGPUSplitFactorize)(x,A,b,update_matrix=false;kwargs...)
    version = b isa CuArray ? CUDADevice() : CPU()
    copyto!(x, b)
    wgs = workgroupsize(version,p.nfacts)
    wait(version, ldiv!_kernel(version)(A,x,p.len,p.nfacts;
                                        ndrange=p.nfacts,
                                        dependencies=Event(version),
                                        workgroupsize=wgs))
    return nothing
end

function (p::LinSolveGPUSplitFactorize)(::Type{Val{:init}},f,u0_prototype)
    LinSolveGPUSplitFactorize(size(u0_prototype)...)
end

@kernel function ldiv!_kernel(W,u,@Const(len),@Const(nfacts))
    i = @index(Global, Linear)
    section = 1 + ((i-1)*len) : (i*len)
    _W = @inbounds @view(W[:, :, i])
    _u = @inbounds @view u[section]
    naivesolve!(_W, _u, len)
end

function __printjac(A, ii)
    @cuprintf "[%d, %d]\n" ii.offset1 ii.stride2
    @cuprintf "%d %d %d\n%d %d %d\n%d %d %d\n" ii[1, 1] ii[1, 2] ii[1, 3] ii[2, 1] ii[2, 2] ii[2, 3] ii[3, 1] ii[3, 2] ii[3, 3]
    @cuprintf "%2.2f %2.2f %2.2f\n%2.2f %2.2f %2.2f\n%2.2f %2.2f %2.2f" A[ii[1, 1]] A[ii[1, 2]] A[ii[1, 3]] A[ii[2, 1]] A[ii[2, 2]] A[ii[2, 3]] A[ii[3, 1]] A[ii[3, 2]] A[ii[3, 3]]
end

function generic_lufact!(A::AbstractMatrix{T}, minmn) where {T}
    m = n = minmn
    #@cuprintf "\n\nbefore lufact!\n"
    #__printjac(A, ii)
    #@cuprintf "\n"
    @inbounds for k = 1:minmn
        #@cuprintf "inner factorization loop\n"
        # Scale first column
        Akkinv = inv(A[k,k])
        for i = k+1:m
            #@cuprintf "L\n"
            A[i,k] *= Akkinv
        end
        # Update the rest
        for j = k+1:n, i = k+1:m
            #@cuprintf "U\n"
            A[i,j] -= A[i,k]*A[k,j]
        end
    end
    #@cuprintf "after lufact!"
    #__printjac(A, ii)
    #@cuprintf "\n\n\n"
    return nothing
end

struct MyL{T} # UnitLowerTriangular
    data::T
end
struct MyU{T} # UpperTriangular
    data::T
end

function naivesub!(A::MyU, b::AbstractVector, n)
    x = b
    @inbounds for j in n:-1:1
        xj = x[j] = A.data[j,j] \ b[j]
        for i in j-1:-1:1 # counterintuitively 1:j-1 performs slightly better
            b[i] -= A.data[i,j] * xj
        end
    end
    return nothing
end
function naivesub!(A::MyL, b::AbstractVector, n)
    x = b
    @inbounds for j in 1:n
        xj = x[j]
        for i in j+1:n
            b[i] -= A.data[i,j] * xj
        end
    end
    return nothing
end

function naivesolve!(A::AbstractMatrix, x::AbstractVector, n)
    naivesub!(MyL(A), x, n)
    naivesub!(MyU(A), x, n)
    return nothing
end

function solve_batch(prob,alg,ensemblealg::EnsembleThreads,II,pmap_batch_size;kwargs...)

  if length(II) == 1 || Threads.nthreads() == 1
    return SciMLBase.solve_batch(prob,alg,EnsembleSerial(),II,pmap_batch_size;kwargs...)
  end

  if typeof(prob.prob) <: SciMLBase.AbstractJumpProblem && length(II) != 1
    probs = [deepcopy(prob.prob) for i in 1:Threads.nthreads()]
  else
    probs = prob.prob
  end

  #
  batch_size = length(II)÷(Threads.nthreads()-1)

  batch_data = tmap(1:(Threads.nthreads()-1)) do i
    if i == Threads.nthreads()-1
      I_local = II[(batch_size*(i-1)+1):end]
    else
      I_local = II[(batch_size*(i-1)+1):(batch_size*i)]
    end
    SciMLBase.solve_batch(prob,alg,EnsembleSerial(),I_local,pmap_batch_size;kwargs...)
  end
  SciMLBase.tighten_container_eltype(batch_data)
end

function tmap(f,args...)
  batch_data = Vector{Core.Compiler.return_type(f,Tuple{typeof.(getindex.(args,1))...})}(undef,length(args[1]))
  Threads.@threads for i in 1:length(args[1])
      batch_data[i] = f(getindex.(args,i)...)
  end
  reduce(vcat,batch_data)
end

export EnsembleCPUArray, EnsembleGPUArray, LinSolveGPUSplitFactorize

end # module
