module DiffEqGPU

using GPUifyLoops, CuArrays, CUDAnative, DiffEqBase
function gpu_kernel(f,du,u,p,t)
    @loop for i in (1:size(u,2); (blockIdx().x-1) * blockDim().x + threadIdx().x)
        @views @inbounds f(du[:,i],u[:,i],p,t)
        nothing
    end
    nothing
end

function GPUifyLoops.launch_config(::typeof(gpu_kernel),maxthreads,context,g,f,du,u,args...;kwargs...)
    t = min(maxthreads,size(u,2))
    blocks = ceil(Int,size(u,2)/t)
    (threads=t,blocks=blocks)
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

    prob = ODEProblem(_f,u0,probs[1].tspan,p)
    sol  = solve(prob,alg; kwargs...)

    us = Array.(sol.u)
    solus = [[us[i][:,j] for i in 1:length(us)] for j in 1:length(probs)]
    [DiffEqBase.build_solution(probs[i],alg,sol.t,solus[i]) for i in 1:length(probs)]
end

export EnsembleCPUArray, EnsembleGPUArray

end # module
