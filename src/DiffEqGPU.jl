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
                 num_monte, batch_size = num_monte/2, kwargs...)

    num_batches = num_monte ÷ batch_size + 1

    time = @elapsed begin
        sols = map(1:num_batches) do i
            if i == num_batches
              I = (batch_size*(i-1)+1):num_monte
            else
              I = (batch_size*(i-1)+1):batch_size*i
            end
            batch_solve(monteprob,I)
        end
    end

    DiffEqBase.MonteCarloSolution(hcat(sols...),time,true)
end

function batch_solve(monteprob,I)
    probs = [monteprob.prob_func(deepcopy(monteprob.prob),i,1) for i in I]
    @assert all(p->p.tspan == probs[1].tspan,probs)
    #@assert all(p->p.f === probs[1].f,probs)

    if montealg isa MonteGPUArray
        u0 = CuArray(hcat([probs[i].u0 for i in I]...))
        p  = CuArray(hcat([probs[i].p  for i in I]...))
    elseif montealg isa MonteCPUArray
        u0 = hcat([probs[i].u0 for i in I]...)
        p  = hcat([probs[i].p  for i in I]...)
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
    solus = [[us[i][:,j] for i in 1:length(us)] for j in I]
    [DiffEqBase.build_solution(probs[i],alg,sol.t,solus[i]) for i in I]
end

export MonteCPUArray, MonteGPUArray

end # module
