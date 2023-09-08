function SciMLBase.__solve(ensembleprob::SciMLBase.AbstractEnsembleProblem,
    alg::Union{SciMLBase.DEAlgorithm, Nothing,
        DiffEqGPU.GPUODEAlgorithm, DiffEqGPU.GPUSDEAlgorithm},
    ensemblealg::Union{EnsembleArrayAlgorithm,
        EnsembleKernelAlgorithm};
    trajectories, batch_size = trajectories,
    unstable_check = (dt, u, p, t) -> false, adaptive = true,
    kwargs...)
    if trajectories == 1
        return SciMLBase.__solve(ensembleprob, alg, EnsembleSerial(); trajectories = 1,
            kwargs...)
    end

    cpu_trajectories = ((ensemblealg isa EnsembleGPUArray ||
                         ensemblealg isa EnsembleGPUKernel) &&
                        ensembleprob.reduction === SciMLBase.DEFAULT_REDUCTION) &&
                       (haskey(kwargs, :callback) ? kwargs[:callback] === nothing : true) ?
                       round(Int, trajectories * ensemblealg.cpu_offload) : 0
    gpu_trajectories = trajectories - cpu_trajectories

    num_batches = gpu_trajectories ÷ batch_size
    num_batches * batch_size != gpu_trajectories && (num_batches += 1)

    if cpu_trajectories != 0 && ensembleprob.reduction === SciMLBase.DEFAULT_REDUCTION
        cpu_II = (gpu_trajectories + 1):trajectories
        _alg = if typeof(alg) <: GPUODEAlgorithm
            if adaptive == false
                cpu_alg[typeof(alg)][1]
            else
                cpu_alg[typeof(alg)][2]
            end
        elseif typeof(alg) <: GPUSDEAlgorithm
            if adaptive == false
                SimpleEM()
            else
                error("Adaptive EM is not supported yet.")
            end
        else
            alg
        end

        function f()
            SciMLBase.solve_batch(ensembleprob, _alg, EnsembleThreads(), cpu_II, nothing;
                kwargs...)
        end

        cpu_sols = Channel{Core.Compiler.return_type(f, Tuple{})}(1)
        t = @task begin
            put!(cpu_sols, f())
        end
        schedule(t)
    end

    if num_batches == 1 && ensembleprob.reduction === SciMLBase.DEFAULT_REDUCTION
        time = @elapsed sol = batch_solve(ensembleprob, alg, ensemblealg,
            1:gpu_trajectories, adaptive;
            unstable_check = unstable_check, kwargs...)
        if cpu_trajectories != 0
            wait(t)
            sol = vcat(sol, take!(cpu_sols))
        end
        return SciMLBase.EnsembleSolution(sol, time, true)
    end

    converged::Bool = false
    u = ensembleprob.u_init === nothing ?
        similar(batch_solve(ensembleprob, alg, ensemblealg, 1:batch_size, adaptive;
            unstable_check = unstable_check, kwargs...), 0) :
        ensembleprob.u_init

    if nprocs() == 1
        # While pmap works, this makes much better error messages.
        time = @elapsed begin
            sols = map(1:num_batches) do i
                if i == num_batches
                    I = (batch_size * (i - 1) + 1):gpu_trajectories
                else
                    I = (batch_size * (i - 1) + 1):(batch_size * i)
                end
                batch_data = batch_solve(ensembleprob, alg, ensemblealg, I, adaptive;
                    unstable_check = unstable_check, kwargs...)
                if ensembleprob.reduction !== SciMLBase.DEFAULT_REDUCTION
                    u, _ = ensembleprob.reduction(u, batch_data, I)
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
                    I = (batch_size * (i - 1) + 1):gpu_trajectories
                else
                    I = (batch_size * (i - 1) + 1):(batch_size * i)
                end
                x = batch_solve(ensembleprob, alg, ensemblealg, I, adaptive;
                    unstable_check = unstable_check, kwargs...)
                yield()
                if ensembleprob.reduction !== SciMLBase.DEFAULT_REDUCTION
                    u, _ = ensembleprob.reduction(u, x, I)
                else
                    x
                end
            end
        end
    end

    if ensembleprob.reduction === SciMLBase.DEFAULT_REDUCTION
        if cpu_trajectories != 0
            wait(t)
            sols = vcat(reduce(vcat, vec.(sols)), take!(cpu_sols))
        else
            sols = reduce(vcat, sols)
        end
        SciMLBase.EnsembleSolution(sols, time, true)
    else
        SciMLBase.EnsembleSolution(sols[end], time, true)
    end
end

function batch_solve(ensembleprob, alg,
    ensemblealg::Union{EnsembleArrayAlgorithm, EnsembleKernelAlgorithm}, I,
    adaptive;
    kwargs...)
    if ensembleprob.safetycopy
        probs = map(I) do i
            ensembleprob.prob_func(deepcopy(ensembleprob.prob), i, 1)
        end
    else
        probs = map(I) do i
            ensembleprob.prob_func(ensembleprob.prob, i, 1)
        end
    end
    @assert !isempty(I)
    #@assert all(p->p.f === probs[1].f,probs)

    if ensemblealg isa EnsembleGPUKernel
        # Using inner saveat requires all of them to be of same size,
        # because the dimension of CuMatrix is decided by it.
        # The columns of it are accessed at each thread.
        if !all(Base.Fix2((prob1, prob2) -> isequal(prob1.tspan, prob2.tspan),
                probs[1]),
            probs)
            if !iszero(ensemblealg.cpu_offload)
                error("Different time spans in an Ensemble Simulation with CPU offloading is not supported yet.")
            end
            if get(probs[1].kwargs, :saveat, nothing) === nothing && !adaptive &&
               get(kwargs, :save_everystep, true)
                error("Using different time-spans require either turning off save_everystep or using saveat. If using saveat, it should be of same length across the ensemble.")
            end
            if !all(Base.Fix2((prob1, prob2) -> isequal(sizeof(get(prob1.kwargs, :saveat,
                            nothing)),
                        sizeof(get(prob2.kwargs, :saveat,
                            nothing))), probs[1]),
                probs)
                error("Using different saveat in EnsembleGPUKernel requires all of them to be of same length. Use saveats of same size only.")
            end
        end

        if alg isa Union{GPUODEAlgorithm, GPUSDEAlgorithm}
            # Get inner saveat if global one isn't specified
            _saveat = get(probs[1].kwargs, :saveat, nothing)
            saveat = _saveat === nothing ? get(kwargs, :saveat, nothing) : _saveat
            solts, solus = batch_solve_up_kernel(ensembleprob, probs, alg, ensemblealg, I,
                adaptive; saveat = saveat, kwargs...)
            [begin
                ts = @view solts[:, i]
                us = @view solus[:, i]
                sol_idx = findlast(x -> x != probs[i].tspan[1], ts)
                if sol_idx === nothing
                    @error "No solution found" tspan=probs[i].tspan[1] ts
                    error("Batch solve failed")
                end
                @views ensembleprob.output_func(SciMLBase.build_solution(probs[i],
                        alg,
                        ts[1:sol_idx],
                        us[1:sol_idx],
                        k = nothing,
                        stats = nothing,
                        calculate_error = false,
                        retcode = sol_idx !=
                                  length(ts) ?
                                  ReturnCode.Terminated :
                                  ReturnCode.Success),
                    i)[1]
            end
             for i in eachindex(probs)]

        else
            error("We don't have solvers implemented for this algorithm yet")
        end
    else
        u0 = reduce(hcat, Array(probs[i].u0) for i in 1:length(I))

        if !all(Base.Fix2((prob1, prob2) -> isequal(prob1.tspan, prob2.tspan),
                probs[1]),
            probs)

            # Requires prob.p to be isbits otherwise it wouldn't work with ParamWrapper
            @assert all(prob -> isbits(prob.p), probs)

            # Remaking the problem to normalize time span values..."
            p = reduce(hcat,
                ParamWrapper(probs[i].p, probs[i].tspan)
                for i in 1:length(I))

            # Change the tspan of first problem to (0,1)
            orig_prob = probs[1]
            probs[1] = remake(probs[1];
                tspan = (zero(probs[1].tspan[1]), one(probs[1].tspan[2])))

            sol, solus = batch_solve_up(ensembleprob, probs, alg, ensemblealg, I,
                u0, p; adaptive = adaptive, kwargs...)

            probs[1] = orig_prob

            [ensembleprob.output_func(SciMLBase.build_solution(probs[i], alg,
                    map(t -> probs[i].tspan[1] +
                             (probs[i].tspan[2] -
                              probs[i].tspan[1]) *
                             t, sol.t), solus[i],
                    stats = sol.stats,
                    retcode = sol.retcode), i)[1]
             for i in 1:length(probs)]
        else
            p = reduce(hcat,
                probs[i].p isa AbstractArray ? Array(probs[i].p) : probs[i].p
                for i in 1:length(I))
            sol, solus = batch_solve_up(ensembleprob, probs, alg, ensemblealg, I, u0, p;
                adaptive = adaptive, kwargs...)
            [ensembleprob.output_func(SciMLBase.build_solution(probs[i], alg, sol.t,
                    solus[i],
                    stats = sol.stats,
                    retcode = sol.retcode), i)[1]
             for i in 1:length(probs)]
        end
    end
end

function batch_solve_up_kernel(ensembleprob, probs, alg, ensemblealg, I, adaptive;
    kwargs...)
    _callback = CallbackSet(generate_callback(probs[1], length(I), ensemblealg; kwargs...))

    _callback = CallbackSet(convert.(DiffEqGPU.GPUDiscreteCallback,
            _callback.discrete_callbacks)...,
        convert.(DiffEqGPU.GPUContinuousCallback,
            _callback.continuous_callbacks)...)

    dev = ensemblealg.dev
    probs = adapt(dev, probs)

    #Adaptive version only works with saveat
    if adaptive
        ts, us = vectorized_asolve(probs, ensembleprob.prob, alg;
            kwargs..., callback = _callback)
    else
        ts, us = vectorized_solve(probs, ensembleprob.prob, alg;
            kwargs..., callback = _callback)
    end
    solus = Array(us)
    solts = Array(ts)
    (solts, solus)
end

function batch_solve_up(ensembleprob, probs, alg, ensemblealg, I, u0, p; kwargs...)
    if ensemblealg isa EnsembleGPUArray
        backend = ensemblealg.backend
        u0 = adapt(backend, u0)
        p = adapt(backend, p)
    end

    len = length(probs[1].u0)

    if SciMLBase.has_jac(probs[1].f)
        if ensemblealg isa EnsembleGPUArray
            backend = ensemblealg.backend
            jac_prototype = allocate(backend, Float32, (len, len, length(I)))
            fill!(jac_prototype, 0.0)
        else
            jac_prototype = zeros(Float32, len, len, length(I))
        end

        if probs[1].f.colorvec !== nothing
            colorvec = repeat(probs[1].f.colorvec, length(I))
        else
            colorvec = repeat(1:length(probs[1].u0), length(I))
        end
    else
        jac_prototype = nothing
        colorvec = nothing
    end

    _callback = generate_callback(probs[1], length(I), ensemblealg; kwargs...)
    prob = generate_problem(probs[1], u0, p, jac_prototype, colorvec)

    if hasproperty(alg, :linsolve)
        _alg = remake(alg, linsolve = LinSolveGPUSplitFactorize(len, -1))
    else
        _alg = alg
    end

    sol = solve(prob, _alg; kwargs..., callback = _callback, merge_callbacks = false,
        internalnorm = diffeqgpunorm)

    us = Array.(sol.u)
    solus = [[@view(us[i][:, j]) for i in 1:length(us)] for j in 1:length(probs)]
    (sol, solus)
end

function seed_duals(x::Matrix{V}, ::Type{T},
    ::ForwardDiff.Chunk{N} = ForwardDiff.Chunk(@view(x[:, 1]),
        typemax(Int64))) where {V, T,
    N}
    seeds = ForwardDiff.construct_seeds(ForwardDiff.Partials{N, V})
    duals = [ForwardDiff.Dual{T}(x[i, j], seeds[i])
             for i in 1:size(x, 1), j in 1:size(x, 2)]
end

function extract_dus(us)
    jsize = size(us[1], 1), ForwardDiff.npartials(us[1][1])
    utype = typeof(ForwardDiff.value(us[1][1]))
    map(1:size(us[1], 2)) do k
        map(us) do u
            du_i = zeros(utype, jsize)
            for i in size(u, 1)
                du_i[i, :] = ForwardDiff.partials(u[i, k])
            end
            du_i
        end
    end
end

struct DiffEqGPUAdjTag end

function ChainRulesCore.rrule(::typeof(batch_solve_up), ensembleprob, probs, alg,
    ensemblealg, I, u0, p; kwargs...)
    pdual = seed_duals(p, DiffEqGPUAdjTag)
    u0 = convert.(eltype(pdual), u0)

    if ensemblealg isa EnsembleGPUArray
        backend = ensemblealg.backend
        u0 = adapt(backend, u0)
        pdual = adapt(backend, pdual)
    end

    len = length(probs[1].u0)

    if SciMLBase.has_jac(probs[1].f)
        if ensemblealg isa EnsembleGPUArray
            backend = ensemblealg.backend
            jac_prototype = allocate(backend, Float32, (len, len, length(I)))
            fill!(jac_prototype, 0.0)
        else
            jac_prototype = zeros(Float32, len, len, length(I))
        end
        if probs[1].f.colorvec !== nothing
            colorvec = repeat(probs[1].f.colorvec, length(I))
        else
            colorvec = repeat(1:length(probs[1].u0), length(I))
        end
    else
        jac_prototype = nothing
        colorvec = nothing
    end

    _callback = generate_callback(probs[1], length(I), ensemblealg)
    prob = generate_problem(probs[1], u0, pdual, jac_prototype, colorvec)

    if hasproperty(alg, :linsolve)
        _alg = remake(alg, linsolve = LinSolveGPUSplitFactorize(len, -1))
    else
        _alg = alg
    end

    sol = solve(prob, _alg; kwargs..., callback = _callback, merge_callbacks = false,
        internalnorm = diffeqgpunorm)

    us = Array.(sol.u)
    solus = [[ForwardDiff.value.(@view(us[i][:, j])) for i in 1:length(us)]
             for j in 1:length(probs)]

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
        (ntuple(_ -> NoTangent(), 7)..., Array(VectorOfArray(adj)))
    end
    (sol, solus), batch_solve_up_adjoint
end

function solve_batch(prob, alg, ensemblealg::EnsembleThreads, II, pmap_batch_size;
    kwargs...)
    if length(II) == 1 || Threads.nthreads() == 1
        return SciMLBase.solve_batch(prob, alg, EnsembleSerial(), II, pmap_batch_size;
            kwargs...)
    end

    if typeof(prob.prob) <: SciMLBase.AbstractJumpProblem && length(II) != 1
        probs = [deepcopy(prob.prob) for i in 1:Threads.nthreads()]
    else
        probs = prob.prob
    end

    #
    batch_size = length(II) ÷ (Threads.nthreads() - 1)

    batch_data = tmap(1:(Threads.nthreads() - 1)) do i
        if i == Threads.nthreads() - 1
            I_local = II[(batch_size * (i - 1) + 1):end]
        else
            I_local = II[(batch_size * (i - 1) + 1):(batch_size * i)]
        end
        SciMLBase.solve_batch(prob, alg, EnsembleSerial(), I_local, pmap_batch_size;
            kwargs...)
    end
    SciMLBase.tighten_container_eltype(batch_data)
end

function tmap(f, args...)
    batch_data = Vector{Core.Compiler.return_type(f, Tuple{typeof.(getindex.(args, 1))...})
    }(undef,
        length(args[1]))
    Threads.@threads for i in 1:length(args[1])
        batch_data[i] = f(getindex.(args, i)...)
    end
    reduce(vcat, batch_data)
end
