module DiffEqGPU

using GPUifyLoops, CuArrays, CUDAnative, DiffEqBase, LinearAlgebra

function gpu_kernel(f,du,u,p,t)
    @loop for i in (1:size(u,2); (blockIdx().x-1) * blockDim().x + threadIdx().x)
        @views @inbounds f(du[:,i],u[:,i],p[:,i],t)
        nothing
    end
    nothing
end

function jac_kernel(f,J,u,p,t)
    @loop for i in (0:(size(u,2)-1); (blockIdx().x-1) * blockDim().x + threadIdx().x)
        section = 1 + (i*size(u,1)) : ((i+1)*size(u,1))
        @views @inbounds f(J[section,section],u[:,i+1],p[:,i+1],t)
        nothing
    end
    nothing
end

function discrete_condition_kernel(condition,cur,u,t,p)
    @loop for i in (1:size(u,2); (blockIdx().x-1) * blockDim().x + threadIdx().x)
        @views @inbounds cur[i] = condition(u[:,i],t,FakeIntegrator(u[:,i],t,p[:,i]))
        nothing
    end
    nothing
end

function discrete_affect!_kernel(affect!,cur,u,t,p)
    @loop for i in (1:size(u,2); (blockIdx().x-1) * blockDim().x + threadIdx().x)
        @views @inbounds cur[i] && affect!(FakeIntegrator(u[:,i],t,p[:,i]))
        nothing
    end
    nothing
end

function continuous_condition_kernel(condition,out,u,t,p)
    @loop for i in (1:size(u,2); (blockIdx().x-1) * blockDim().x + threadIdx().x)
        @views @inbounds out[i] = condition(u[:,i],t,FakeIntegrator(u[:,i],t,p[:,i]))
        nothing
    end
    nothing
end

function continuous_affect!_kernel(affect!,event_idx,u,t,p)
    @loop for i in ((event_idx,); (blockIdx().x-1) * blockDim().x + threadIdx().x)
        @views @inbounds affect!(FakeIntegrator(u[:,i],t,p[:,i]))
        nothing
    end
    nothing
end

function GPUifyLoops.launch_config(::typeof(gpu_kernel),maxthreads,context,g,f,du,u,args...;kwargs...)
    t = min(maxthreads,size(u,2))
    blocks = ceil(Int,size(u,2)/t)
    (threads=t,blocks=blocks)
end

struct FakeIntegrator{uType,tType,P}
    u::uType
    t::tType
    p::P
end

abstract type EnsembleArrayAlgorithm <: DiffEqBase.EnsembleAlgorithm end
struct EnsembleCPUArray <: EnsembleArrayAlgorithm end
struct EnsembleGPUArray <: EnsembleArrayAlgorithm end

function DiffEqBase.__solve(ensembleprob::DiffEqBase.AbstractEnsembleProblem,
                 alg::Union{DiffEqBase.DEAlgorithm,Nothing},
                 ensemblealg::EnsembleArrayAlgorithm;
                 trajectories, batch_size = trajectories, kwargs...)

    num_batches = trajectories ÷ batch_size

    num_batches * batch_size != trajectories && (num_batches += 1)
    time = @elapsed begin
        sols = map(1:num_batches) do i
            if i == num_batches
              I = (batch_size*(i-1)+1):trajectories
            else
              I = (batch_size*(i-1)+1):batch_size*i
            end
            batch_solve(ensembleprob,alg,ensemblealg,I;kwargs...)
        end
    end

    DiffEqBase.EnsembleSolution(hcat(sols...),time,true)
end

function batch_solve(ensembleprob,alg,ensemblealg,I;kwargs...)
    probs = [ensembleprob.prob_func(deepcopy(ensembleprob.prob),i,1) for i in I]
    @assert all(p->p.tspan == probs[1].tspan,probs)
    @assert !isempty(I)
    #@assert all(p->p.f === probs[1].f,probs)

    if ensemblealg isa EnsembleGPUArray
        u0 = CuArray(hcat([probs[i].u0 for i in 1:length(probs)]...))
        p  = CuArray(hcat([probs[i].p  for i in 1:length(probs)]...))
    elseif ensemblealg isa EnsembleCPUArray
        u0 = hcat([probs[i].u0 for i in 1:length(probs)]...)
        p  = hcat([probs[i].p  for i in 1:length(probs)]...)
    end

    _f = let f=probs[1].f
        function (du,u,p,t)
            version = u isa CuArray ? CUDA() : CPU()
            @launch version gpu_kernel(f,du,u,p,t)
        end
    end

    if DiffEqBase.has_jac(probs[1].f)
        _jac = let jac=probs[1].f.jac
            function (J,u,p,t)
                version = u isa CuArray ? CUDA() : CPU()
                @launch version jac_kernel(jac,J,u,p,t)
            end
        end
    else
        _jac = nothing
    end

    if probs[1].f.colorvec !== nothing
        colorvec = repeat(probs[1].f.colorvec,length(I))
    else
        colorvec = repeat(1:length(probs[1].u0),length(I))
    end

    if :callback ∉ keys(probs[1].kwargs)
        _callback = nothing
    elseif probs[1].kwargs[:callback] isa DiscreteCallback
        if ensemblealg isa EnsembleGPUArray
            cur = CuArray([false for i in 1:length(probs)])
        else
            cur = [false for i in 1:length(probs)]
        end
        _condition = probs[1].kwargs[:callback].condition
        _affect!   = probs[1].kwargs[:callback].affect!

        condition = function (u,t,integrator)
            version = u isa CuArray ? CUDA() : CPU()
            @launch version discrete_condition_kernel(_condition,cur,u,t,integrator.p)
            any(cur)
        end

        affect! = function (integrator)
            version = integrator.u isa CuArray ? CUDA() : CPU()
            @launch version discrete_affect!_kernel(_affect!,cur,integrator.u,integrator.t,integrator.p)
        end

        _callback = DiscreteCallback(condition,affect!,save_positions=probs[1].kwargs[:callback].save_positions)
    elseif probs[1].kwargs[:callback] isa ContinuousCallback
        _condition   = probs[1].kwargs[:callback].condition
        _affect!     = probs[1].kwargs[:callback].affect!
        _affect_neg! = probs[1].kwargs[:callback].affect_neg!

        condition = function (out,u,t,integrator)
            version = u isa CuArray ? CUDA() : CPU()
            @launch version continuous_condition_kernel(_condition,out,u,t,integrator.p)
            nothing
        end

        affect! = function (integrator,event_idx)
            version = integrator.u isa CuArray ? CUDA() : CPU()
            @launch version continuous_affect!_kernel(_affect!,event_idx,integrator.u,integrator.t,integrator.p)
        end

        affect_neg! = function (integrator,event_idx)
            version = integrator.u isa CuArray ? CUDA() : CPU()
            @launch version continuous_affect!_kernel(_affect_neg!,event_idx,integrator.u,integrator.t,integrator.p)
        end

        _callback = VectorContinuousCallback(condition,affect!,affect_neg!,length(probs),save_positions=probs[1].kwargs[:callback].save_positions)
    end

    f_func = ODEFunction(_f,jac=_jac,colorvec=colorvec)
    prob = ODEProblem(f_func,u0,probs[1].tspan,p;
                      probs[1].kwargs...)
    sol  = solve(prob,alg; callback = _callback, kwargs...)

    us = Array.(sol.u)
    solus = [[us[i][:,j] for i in 1:length(us)] for j in 1:length(probs)]
    [DiffEqBase.build_solution(probs[i],alg,sol.t,solus[i],destats=sol.destats,retcode=sol.retcode) for i in 1:length(probs)]
end

### GPU Factorization

mutable struct LinSolveGPUSplitFactorize{T}
  facts::Array{CuArrays.CUSOLVER.CuQR{T,CuArray{T,2}}}
  len::Int
end
LinSolveGPUSplitFactorize() = LinSolveGPUSplitFactorize(CuArrays.CUSOLVER.CuQR{Float32,CuArray{Float32,2}}[],0)

function (p::LinSolveGPUSplitFactorize)(x,A,b,update_matrix=false;kwargs...)
  version = b isa CuArray ? CUDA() : CPU()
  if update_matrix
    @launch version qr_kernel(p.facts,A)
  end
  if typeof(p.A) <: SuiteSparse.UMFPACK.UmfpackLU || typeof(p.factorization) <: typeof(lu)
    ldiv!(x,p.A,b) # No 2-arg form for SparseArrays!
  else
    x .= b
    @launch version ldiv!_kernel(p.facts,x,p.len)
  end
end
function (p::LinSolveGPUSplitFactorize)(::Type{Val{:init}},f,u0_prototype)
  LinSolveGPUSplitFactorize(Array{CuArrays.CUSOLVER.CuQR{eltype(u0_prototype),CuArray{eltype(u0_prototype),2}}}(undef,size(u0_prototype,2)),size(u0_prototype,1))
end

function qr_kernel(facts,W,len)
    @loop for i in (0:length(facts)-1; (blockIdx().x-1) * blockDim().x + threadIdx().x)
        section = 1 + (i*len) : ((i+1)*len)
        facts[i] = qr!(W[])
        nothing
    end
    nothing
end

function ldiv!_kernel(facts,W,len)
    @loop for i in (0:length(facts)-1; (blockIdx().x-1) * blockDim().x + threadIdx().x)
        @views ldiv!(facts[i],x[(i*len+1):((i+1)*len)])
        nothing
    end
    nothing
end

export EnsembleCPUArray, EnsembleGPUArray, LinSolveGPUSplitFactorize

end # module
