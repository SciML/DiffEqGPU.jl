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
        u0 = CuArray(hcat([probs[i].u0 for i in 1:length(I)]...))
        p  = CuArray(hcat([probs[i].p  for i in 1:length(I)]...))
    elseif ensemblealg isa EnsembleCPUArray
        u0 = hcat([probs[i].u0 for i in 1:length(I)]...)
        p  = hcat([probs[i].p  for i in 1:length(I)]...)
    end

    _f = let f=probs[1].f.f
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

    if probs[1].f.colorvec !== nothing
        jac_prototype = CuArray(repeat(probs[1].f.jac_prototype,length(I)))
    else
        jac_prototype = cu(zeros(Float32,length(probs[1].u0)*length(I),length(probs[1].u0)))
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

    #=
    internalnorm(u::CuArray,t) = sqrt(maximum(reduce((x,y)->x^2 + y^2, u, dim=1))/size(u0,1))
    internalnorm(u::Union{AbstractFloat,Complex},t) = abs(u)
    =#

    f_func = ODEFunction(_f,jac=_jac,
                        #colorvec=colorvec,
                        #jac_prototype = jac_prototype,
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
    if update_matrix
        #println("\nbefore")
        #Base.print_matrix(stdout, Array(A))
        #flush(stdout)
        #@show p.len,p.nfacts
        @launch version qr_kernel(A,p.len,p.nfacts)
        #println("\nafter")
        #Base.print_matrix(stdout, Array(A))
    end
    copyto!(x, b)
    @launch version ldiv!_kernel(A,x,p.len,p.nfacts)
    return nothing
end

function (p::LinSolveGPUSplitFactorize)(::Type{Val{:init}},f,u0_prototype)
    LinSolveGPUSplitFactorize(size(u0_prototype)...)
end

struct SimpleView
    offset1::Int
    stride2::Int # stride(A, 2)
end

@inline Base.getindex(sv::SimpleView, i::Integer, j::Integer) = sv.offset1 + i + sv.stride2 * (j-Int(1) + sv.offset1)
@inline Base.getindex(sv::SimpleView, i::Integer) = sv.offset1 + i

function qr_kernel(W,len,nfacts)
    stride2 = size(W, 1)
    @loop for i in (0:(nfacts-1); (blockIdx().x-1) * blockDim().x + threadIdx().x - 1)
        offset = i*len
        sv = SimpleView(offset, stride2)
        #@cuprintf "\n\nouter factorization loop\n"
        generic_lufact!(W, sv, len)
        nothing
    end
    return nothing
end

function ldiv!_kernel(W,x,len,nfacts)
    stride2 = size(W, 1)
    @loop for i in (0:(nfacts-1); (blockIdx().x-1) * blockDim().x + threadIdx().x - 1)
        offset = i*len
        sv = SimpleView(offset, stride2)
        naivesolve!(W, x, sv, len)
        nothing
    end
    return nothing
end

function GPUifyLoops.launch_config(::typeof(qr_kernel),
                                           maxthreads,context,g,W,len,nfacts;
                                           kwargs...)
    t = min(maxthreads,nfacts)
    blocks = ceil(Int,nfacts/t)
    (threads=t,blocks=blocks)
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

function generic_lufact!(A::AbstractMatrix{T}, ii, minmn) where {T}
    m = n = minmn
    #@cuprintf "\n\nbefore lufact!\n"
    #__printjac(A, ii)
    #@cuprintf "\n"
    @inbounds for k = 1:minmn
        #@cuprintf "inner factorization loop\n"
        # Scale first column
        Akkinv = inv(A[ii[k,k]])
        for i = k+1:m
            #@cuprintf "L\n"
            A[ii[i,k]] *= Akkinv
        end
        # Update the rest
        for j = k+1:n, i = k+1:m
            #@cuprintf "U\n"
            A[ii[i,j]] -= A[ii[i,k]]*A[ii[k,j]]
        end
    end
    #@cuprintf "after lufact!"
    #__printjac(A, ii)
    #@cuprintf "\n\n\n"
    return nothing
end

function naivesub!(A::UpperTriangular, b::AbstractVector, ii, n)
    x = b
    @inbounds for j in n:-1:1
        xj = x[ii[j]] = A.data[ii[j,j]] \ b[ii[j]]
        for i in j-1:-1:1 # counterintuitively 1:j-1 performs slightly better
            b[ii[i]] -= A.data[ii[i,j]] * xj
        end
    end
    return nothing
end
function naivesub!(A::UnitLowerTriangular, b::AbstractVector, ii, n)
    x = b
    @inbounds for j in 1:n
        xj = x[ii[j]]
        for i in j+1:n
            b[ii[i]] -= A.data[ii[i,j]] * xj
        end
    end
    return nothing
end

function naivesolve!(A::AbstractMatrix, x::AbstractVector, ii, n)
    naivesub!(UnitLowerTriangular(A), x, ii, n)
    naivesub!(UpperTriangular(A), x, ii, n)
    return nothing
end

function LinearAlgebra.checksquare(A::CUDAnative.CuDeviceArray)
    m,n = size(A)
    m
end

export EnsembleCPUArray, EnsembleGPUArray, LinSolveGPUSplitFactorize

end # module
