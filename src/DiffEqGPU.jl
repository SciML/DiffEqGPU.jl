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

abstract type MonteArrayAlgorithm <: DiffEqBase.MonteCarloAlgorithm end
struct MonteCPUArray <: MonteArrayAlgorithm end
struct MonteGPUArray <: MonteArrayAlgorithm end

function DiffEqBase.__solve(monteprob::DiffEqBase.AbstractMonteCarloProblem,
                 alg::Union{DiffEqBase.DEAlgorithm,Nothing},
                 montealg::MonteArrayAlgorithm;
                 num_monte, batch_size = num_monte,
                 pmap_batch_size = batch_size÷100 > 0 ? batch_size÷100 : 1, kwargs...)

    probs = [monteprob.prob_func(deepcopy(monteprob.prob),i,1) for i in 1:num_monte]
    @assert all(p->p.tspan == probs[1].tspan,probs)
    #@assert all(p->p.f === probs[1].f,probs)

    if montealg isa MonteGPUArray
        u0 = CuArray(hcat([probs[i].u0 for i in 1:num_monte]...))
        p = CuArray(hcat([probs[i].p for i in 1:num_monte]...))
    elseif montealg isa MonteCPUArray
        u0 = hcat([probs[i].u0 for i in 1:num_monte]...)
        p = hcat([probs[i].p for i in 1:num_monte]...)
    end

    _f = let f=probs[1].f
        function (du,u,p,t)
            version = u isa CuArray ? CUDA() : CPU()
            @launch version gpu_kernel(f,du,u,p,t)
        end
    end

    prob = ODEProblem(_f,u0,probs[1].tspan,p)
    solve(prob,alg; kwargs...)
end

export MonteCPUArray, MonteGPUArray

end # module
