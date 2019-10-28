module DiffEqGPU

using GPUifyLoops, CuArrays, CUDAnative, DiffEqBase, LinearAlgebra
using CUDAdrv: CuPtr, CU_NULL, Mem, CuDefaultStream
using CuArrays: CUBLAS

function gpu_kernel(f,du,u,p,t)
    @loop for i in (1:size(u,2); (blockIdx().x-1) * blockDim().x + threadIdx().x)
        @views @inbounds f(du[:,i],u[:,i],p[:,i],t)
        nothing
    end
    nothing
end

function jac_kernel(f,J,u,p,t)
    @loop for i in (0:(size(u,2)-1); (blockIdx().x-1) * blockDim().x + threadIdx().x - 1)
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

function GPUifyLoops.launch_config(::Union{typeof(gpu_kernel),
                                           typeof(jac_kernel),
                                           typeof(discrete_condition_kernel),
                                           typeof(discrete_affect!_kernel),
                                           typeof(continuous_condition_kernel),
                                           typeof(continuous_affect!_kernel)},
                                           maxthreads,context,g,f,du,u,args...;
                                           kwargs...)
    t = min(maxthreads,size(u,2))
    blocks = ceil(Int,size(u,2)/t)
    (threads=t,blocks=blocks)
end

function W_kernel(jac, W, u, p, gamma, t)
    len = size(u,1)
    @loop for i in (1:size(u,2); (blockIdx().x-1) * blockDim().x + threadIdx().x)
        _W = @inbounds @view(W[:, :, i])
        @views @inbounds jac(_W,u[:,i],p[:,i],t)
        @inbounds for i in eachindex(_W)
            _W[i] = gamma*_W[i]
        end
        _one = one(eltype(_W))
        @inbounds for i in 1:len
            _W[i, i] = _W[i, i] - _one
        end
        nothing
    end
    return nothing
end

function Wt_kernel(jac, W, u, p, gamma, t)
    len = size(u,1)
    @loop for i in (1:size(u,2); (blockIdx().x-1) * blockDim().x + threadIdx().x)
        _W = @inbounds @view(W[:, :, i])
        @views @inbounds jac(_W,u[:,i],p[:,i],t)
        @inbounds for i in 1:len
            _W[i, i] = -inv(gamma) + _W[i, i]
        end
        nothing
    end
    return nothing
end

function GPUifyLoops.launch_config(::Union{typeof(W_kernel),typeof(Wt_kernel)},
                                           maxthreads,context,g,jac,W,u,args...;
                                           kwargs...)
    t = min(maxthreads,size(u,2))
    blocks = ceil(Int,size(u,2)/t)
    (threads=t,blocks=blocks)
end

function cuda_lufact!(W)
    T = eltype(W)
    batchsize = size(W, 3)
    batch = size(W)[1:2]
    @assert batch[1] == batch[2]
    A = Vector{CuPtr{T}}(undef, batchsize)
    ptr = Base.unsafe_convert(CuPtr{T}, Base.cconvert(CuPtr{T}, W))
    for i in 1:batchsize
        A[i] = ptr + (i-1) * prod(batch) * sizeof(T)
    end
    B = CuVector{eltype(A)}(undef, batchsize)
    Mem.copy!(B.buf, pointer(A), sizeof(A); async=true, stream=CuDefaultStream())
    lda = max(1, stride(W,2))
    info = CuArray{Cint}(undef, length(A))
    CUBLAS.cublasSgetrfBatched(CUBLAS.handle(), batch[1], B, lda, CU_NULL, info, batchsize)
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

    len = length(probs[1].u0)
    if ensemblealg isa EnsembleGPUArray
        u0 = CuArray(hcat([probs[i].u0 for i in 1:length(I)]...))
        p  = CuArray(hcat([probs[i].p  for i in 1:length(I)]...))
        jac_prototype = cu(zeros(Float32,len,len,length(I)))
    elseif ensemblealg isa EnsembleCPUArray
        u0 = hcat([probs[i].u0 for i in 1:length(I)]...)
        p  = hcat([probs[i].p  for i in 1:length(I)]...)
        jac_prototype = zeros(len,len,length(I))
    end

    _f = let f=probs[1].f.f
        function (du,u,p,t)
            version = u isa CuArray ? CUDA() : CPU()
            @launch version gpu_kernel(f,du,u,p,t)
        end
    end

    if DiffEqBase.has_jac(probs[1].f)
        _Wfact! = let jac=probs[1].f.jac
            function (W,u,p,gamma,t)
                iscuda = u isa CuArray
                version = iscuda ? CUDA() : CPU()
                @launch version W_kernel(jac, W, u, p, gamma, t)
                iscuda ? cuda_lufact!(W) : cpu_lufact!(W)
            end
        end
        _Wfact!_t = let jac=probs[1].f.jac
            function (W,u,p,gamma,t)
                iscuda = u isa CuArray
                version = iscuda ? CUDA() : CPU()
                @launch version Wt_kernel(jac, W, u, p, gamma, t)
                iscuda ? cuda_lufact!(W) : cpu_lufact!(W)
            end
        end
    else
        _Wfact! = nothing
        _Wfact!_t = nothing
    end

    if DiffEqBase.has_tgrad(probs[1].f)
        _tgrad = let tgrad=probs[1].f.tgrad
            function (J,u,p,t)
                version = u isa CuArray ? CUDA() : CPU()
                @launch version gpu_kernel(tgrad,J,u,p,t)
            end
        end
    else
        _tgrad = nothing
    end

    if probs[1].f.colorvec !== nothing
        colorvec = repeat(probs[1].f.colorvec,length(I))
    else
        colorvec = repeat(1:length(probs[1].u0),length(I))
    end

    #=
    if probs[1].f.colorvec !== nothing
        jac_prototype = CuArray(repeat(probs[1].f.jac_prototype,length(I)))
    else
        jac_prototype = cu(zeros(Float32,length(probs[1].u0)^2,length(I)))
    end
    =#

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

    internalnorm(u::CuArray,t) = sqrt(maximum(sum(abs2, u, dims=1)/size(u, 1)))
    internalnorm(u::Union{AbstractFloat,Complex},t) = abs(u)

    f_func = ODEFunction(_f,Wfact = _Wfact!,
                        Wfact_t = _Wfact!_t,
                        #colorvec=colorvec,
                        jac_prototype = jac_prototype,
                        tgrad=_tgrad)
    prob = ODEProblem(f_func,u0,probs[1].tspan,p;
                      probs[1].kwargs...)
    sol  = solve(prob,alg; callback = _callback,
                 #internalnorm=internalnorm,
                 kwargs...)

    us = Array.(sol.u)
    solus = [[us[i][:,j] for i in 1:length(us)] for j in 1:length(probs)]
    [ensembleprob.output_func(DiffEqBase.build_solution(probs[i],alg,sol.t,solus[i],destats=sol.destats,retcode=sol.retcode),i)[1] for i in 1:length(probs)]
end

### GPU Factorization

struct LinSolveGPUSplitFactorize
    len::Int
    nfacts::Int
end
LinSolveGPUSplitFactorize() = LinSolveGPUSplitFactorize(0, 0)

function (p::LinSolveGPUSplitFactorize)(x,A,b,update_matrix=false;kwargs...)
    version = b isa CuArray ? CUDA() : CPU()
    copyto!(x, b)
    @launch version ldiv!_kernel(A,x,p.len,p.nfacts)
    return nothing
end

function (p::LinSolveGPUSplitFactorize)(::Type{Val{:init}},f,u0_prototype)
    LinSolveGPUSplitFactorize(size(u0_prototype)...)
end

function ldiv!_kernel(W,u,len,nfacts)
    @loop for i in (1:nfacts; (blockIdx().x-1) * blockDim().x + threadIdx().x)
        section = 1 + ((i-1)*len) : (i*len)
        _W = @inbounds @view(W[:, :, i])
        _u = @inbounds @view u[section]
        naivesolve!(_W, _u, len)
        nothing
    end
    return nothing
end

function GPUifyLoops.launch_config(::typeof(ldiv!_kernel),
                                           maxthreads,context,g,W,x,len,nfacts,
                                           args...;
                                           kwargs...)
    t = min(maxthreads,nfacts)
    blocks = ceil(Int,nfacts/t)
    (threads=t,blocks=blocks)
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

export EnsembleCPUArray, EnsembleGPUArray, LinSolveGPUSplitFactorize

end # module
