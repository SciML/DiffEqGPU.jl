function SciMLBase.__solve(
        ensembleprob::SciMLBase.AbstractEnsembleProblem,
        alg::Union{
            SciMLBase.AbstractDEAlgorithm, Nothing,
            DiffEqGPU.GPUODEAlgorithm, DiffEqGPU.GPUSDEAlgorithm,
        },
        ensemblealg::Union{
            EnsembleArrayAlgorithm,
            EnsembleKernelAlgorithm,
        };
        trajectories, batch_size = trajectories,
        unstable_check = (dt, u, p, t) -> false, adaptive = true,
        seed = nothing,
        rng = nothing,
        rng_func = SciMLBase.default_rng_func,
        kwargs...
    )
    if trajectories == 1
        return SciMLBase.__solve(
            ensembleprob, alg, EnsembleSerial(); trajectories = 1,
            seed, rng, rng_func, kwargs...
        )
    end

    # Pre-generate per-trajectory seeds for reproducibility (matching SciMLBase v3 protocol)
    sim_seeds = (rng !== nothing || seed !== nothing) ?
        SciMLBase.generate_sim_seeds(rng, seed, trajectories) : nothing

    # Bundle ensemble RNG state for passing to SciMLBase.solve_batch (CPU offload path)
    ensemble_rng_state = (;
        sim_seeds,
        _solve_rng_mode = Val(:none),
        rng_func,
        master_rng = rng,
    )

    cpu_trajectories = (
            (
                ensemblealg isa EnsembleGPUArray ||
                ensemblealg isa EnsembleGPUKernel
            ) &&
            ensembleprob.reduction === SciMLBase.DEFAULT_REDUCTION
        ) &&
        (haskey(kwargs, :callback) ? kwargs[:callback] === nothing : true) ?
        round(Int, trajectories * ensemblealg.cpu_offload) : 0
    gpu_trajectories = trajectories - cpu_trajectories

    if alg isa GPUTsit5IController && cpu_trajectories != 0
        throw(ArgumentError("GPUTsit5IController requires cpu_offload = 0."))
    end

    num_batches = gpu_trajectories ÷ batch_size
    num_batches * batch_size != gpu_trajectories && (num_batches += 1)

    cpu_work = if cpu_trajectories != 0 &&
            ensembleprob.reduction === SciMLBase.DEFAULT_REDUCTION
        cpu_II = (gpu_trajectories + 1):trajectories
        _alg = if alg isa GPUODEAlgorithm
            if adaptive == false
                cpu_alg[typeof(alg)][1]
            else
                cpu_alg[typeof(alg)][2]
            end
        elseif alg isa GPUSDEAlgorithm
            if adaptive == false
                SimpleEM()
            else
                error("Adaptive EM is not supported yet.")
            end
        else
            alg
        end

        function f()
            return SciMLBase.solve_batch(
                ensembleprob, _alg, EnsembleThreads(), cpu_II, nothing,
                ensemble_rng_state; kwargs...
            )
        end

        cpu_sols = Channel{Core.Compiler.return_type(f, Tuple{})}(1)
        t = @task begin
            put!(cpu_sols, f())
        end
        schedule(t)
        (; cpu_sols, t)
    else
        nothing
    end

    if num_batches == 1 && ensembleprob.reduction === SciMLBase.DEFAULT_REDUCTION
        time = @elapsed sol = batch_solve(
            ensembleprob, alg, ensemblealg,
            1:gpu_trajectories, adaptive;
            sim_seeds, rng_func, master_rng = rng,
            unstable_check, kwargs...
        )
        if cpu_work !== nothing
            wait(cpu_work.t)
            sol = vcat(sol, take!(cpu_work.cpu_sols))
        end
        return SciMLBase.EnsembleSolution(sol, time, true)
    end

    converged::Bool = false
    u = ensembleprob.u_init === nothing ?
        similar(
            batch_solve(
                ensembleprob, alg, ensemblealg, 1:batch_size, adaptive;
                sim_seeds, rng_func, master_rng = rng,
                unstable_check, kwargs...
            ),
            0
        ) :
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
                batch_data = batch_solve(
                    ensembleprob, alg, ensemblealg, I, adaptive;
                    sim_seeds, rng_func, master_rng = rng,
                    unstable_check, kwargs...
                )
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
                x = batch_solve(
                    ensembleprob, alg, ensemblealg, I, adaptive;
                    sim_seeds, rng_func, master_rng = rng,
                    unstable_check, kwargs...
                )
                yield()
                if ensembleprob.reduction !== SciMLBase.DEFAULT_REDUCTION
                    u, _ = ensembleprob.reduction(u, x, I)
                else
                    x
                end
            end
        end
    end

    return if ensembleprob.reduction === SciMLBase.DEFAULT_REDUCTION
        if cpu_work !== nothing
            wait(cpu_work.t)
            sols = vcat(reduce(vcat, vec.(sols)), take!(cpu_work.cpu_sols))
        else
            sols = reduce(vcat, sols)
        end
        SciMLBase.EnsembleSolution(sols, time, true)
    else
        SciMLBase.EnsembleSolution(sols[end], time, true)
    end
end

function _make_ensemble_context(i, sim_seeds, rng_func, master_rng)
    sim_seed = sim_seeds !== nothing ? sim_seeds[i] : nothing
    pre_ctx = SciMLBase.EnsembleContext(i, 1, 0, sim_seed, nothing, master_rng)
    sim_rng = rng_func(pre_ctx)
    return @set pre_ctx.rng = sim_rng
end

# Enzyme needs numeric fields first and uninlined construction to retain aggregate type information.
struct KernelODEProblem{U, T, IIP, P, F, K, PT} <: SciMLBase.AbstractODEProblem{U, T, IIP}
    p::P
    u0::U
    tspan::T
    f::F
    kwargs::K
    problem_type::PT
end
_kernel_record(prob) = prob
@noinline function _kernel_record(prob::SciMLBase.ImmutableODEProblem{U, T, IIP, P, F, K, PT}) where {U, T, IIP, P, F, K, PT}
    return KernelODEProblem{U, T, IIP, P, F, K, PT}(prob.p, prob.u0, prob.tspan, prob.f, prob.kwargs, prob.problem_type)
end

@noinline function _make_kernel_problem(ensembleprob, i, sim_seeds, rng_func, master_rng)
    ctx = _make_ensemble_context(i, sim_seeds, rng_func, master_rng)
    prob = ensembleprob.safetycopy ? deepcopy(ensembleprob.prob) : ensembleprob.prob
    return make_prob_compatible(ensembleprob.prob_func(prob, ctx))
end

# Enzyme needs reference storage for host aggregate shadows, but primal solves do not.
@inline _wrap_kernel_host(prob) = within_autodiff() ? Ref(prob) : prob
@inline _unwrap_kernel_host(prob) = prob
@inline _unwrap_kernel_host(prob::Ref) = prob[]

function _prepare_kernel_problems(ensembleprob, backend, I, sim_seeds, rng_func, master_rng)
    first_prob = _make_kernel_problem(ensembleprob, first(I), sim_seeds, rng_func, master_rng)
    first_adapted = _kernel_record(adapt(backend, first_prob))
    first_ref = _wrap_kernel_host(first_prob)
    probs = Vector{typeof(first_ref)}(undef, length(I))
    adapted_probs = Vector{typeof(first_adapted)}(undef, length(I))
    probs[1] = first_ref
    adapted_probs[1] = first_adapted
    # Adapt during construction: a separate broadcast over isbits problems loses Enzyme gradients.
    for j in 2:length(I)
        prob = _make_kernel_problem(ensembleprob, I[j], sim_seeds, rng_func, master_rng)
        probs[j] = _wrap_kernel_host(prob)
        adapted_probs[j] = _kernel_record(adapt(backend, prob))
    end
    return probs, adapted_probs
end

function batch_solve(
        ensembleprob, alg,
        ensemblealg::Union{EnsembleArrayAlgorithm, EnsembleKernelAlgorithm}, I,
        adaptive;
        sim_seeds = nothing,
        rng_func = SciMLBase.default_rng_func,
        master_rng = nothing,
        kwargs...
    )
    @assert !isempty(I)

    return if ensemblealg isa EnsembleGPUKernel
        kernel_probs, adapted_kernel_probs = _prepare_kernel_problems(
            ensembleprob, ensemblealg.dev, I, sim_seeds, rng_func, master_rng
        )
        if !all(
                Base.Fix2(
                    (prob1, prob2) -> isequal(_unwrap_kernel_host(prob1).tspan, _unwrap_kernel_host(prob2).tspan),
                    kernel_probs[1]
                ),
                kernel_probs
            )
            if !iszero(ensemblealg.cpu_offload)
                error("Different time spans in an Ensemble Simulation with CPU offloading is not supported yet.")
            end
            if get(_unwrap_kernel_host(kernel_probs[1]).kwargs, :saveat, nothing) === nothing && !adaptive &&
                    get(kwargs, :save_everystep, true)
                error("Using different time-spans require either turning off save_everystep or using saveat. If using saveat, it should be of same length across the ensemble.")
            end
            if !all(
                    Base.Fix2(
                        (
                            prob1,
                            prob2,
                        ) -> isequal(
                            sizeof(get(_unwrap_kernel_host(prob1).kwargs, :saveat, nothing)),
                            sizeof(get(_unwrap_kernel_host(prob2).kwargs, :saveat, nothing))
                        ),
                        kernel_probs[1]
                    ),
                    kernel_probs
                )
                error("Using different saveat in EnsembleGPUKernel requires all of them to be of same length. Use saveats of same size only.")
            end
        end

        if !(alg isa Union{GPUODEAlgorithm, GPUSDEAlgorithm})
            error("We don't have solvers implemented for this algorithm yet")
        end

        _saveat = get(_unwrap_kernel_host(kernel_probs[1]).kwargs, :saveat, nothing)
        saveat = _saveat === nothing ? get(kwargs, :saveat, nothing) : _saveat
        solts, kernel_solus = batch_solve_up_kernel(
            ensembleprob, kernel_probs, adapted_kernel_probs, alg, ensemblealg, I,
            adaptive; saveat, kwargs...
        )
        [
            begin
                ts = @view solts[:, i]
                us = @view kernel_solus[:, i]
                sol_idx = findlast(x -> x != _unwrap_kernel_host(kernel_probs[i]).tspan[1], ts)
                if sol_idx === nothing
                    @error "No solution found" tspan = _unwrap_kernel_host(kernel_probs[i]).tspan[1] ts
                    error("Batch solve failed")
                end
                @views ensembleprob.output_func(
                    SciMLBase.build_solution(
                        _unwrap_kernel_host(kernel_probs[i]),
                        alg,
                        ts[1:sol_idx],
                        us[1:sol_idx],
                        k = nothing,
                        stats = nothing,
                        calculate_error = false,
                        retcode = sol_idx != length(ts) ? ReturnCode.Terminated :
                            ReturnCode.Success
                    ),
                    _make_ensemble_context(I[i], sim_seeds, rng_func, master_rng)
                )[1]
            end
                for i in eachindex(kernel_probs)
        ]
    else
        if ensembleprob.safetycopy
            probs = map(I) do i
                ctx = _make_ensemble_context(i, sim_seeds, rng_func, master_rng)
                ensembleprob.prob_func(deepcopy(ensembleprob.prob), ctx)
            end
        else
            probs = map(I) do i
                ctx = _make_ensemble_context(i, sim_seeds, rng_func, master_rng)
                ensembleprob.prob_func(ensembleprob.prob, ctx)
            end
        end
        u0 = reduce(hcat, Array(probs[i].u0) for i in 1:length(I))

        if !all(
                Base.Fix2(
                    (prob1, prob2) -> isequal(prob1.tspan, prob2.tspan),
                    probs[1]
                ),
                probs
            )

            # Requires prob.p to be isbits otherwise it wouldn't work with ParamWrapper
            @assert all(prob -> isbits(prob.p), probs)

            # Remaking the problem to normalize time span values..."
            p = reduce(
                hcat,
                ParamWrapper(probs[i].p, probs[i].tspan)
                    for i in 1:length(I)
            )

            # Change the tspan of first problem to (0,1)
            orig_prob = probs[1]
            probs[1] = remake(
                probs[1];
                tspan = (zero(probs[1].tspan[1]), one(probs[1].tspan[2]))
            )

            sol,
                solus = batch_solve_up(
                ensembleprob, probs, alg, ensemblealg, I,
                u0, p; adaptive, kwargs...
            )

            probs[1] = orig_prob

            [
                ensembleprob.output_func(
                    SciMLBase.build_solution(
                        probs[i], alg,
                        map(
                            t -> probs[i].tspan[1] +
                                (
                                probs[i].tspan[2] -
                                    probs[i].tspan[1]
                            ) *
                                t,
                            sol.t
                        ), solus[i],
                        stats = sol.stats,
                        retcode = sol.retcode
                    ),
                    _make_ensemble_context(I[i], sim_seeds, rng_func, master_rng)
                )[1]
                    for i in 1:length(probs)
            ]
        else
            p = reduce(
                hcat,
                probs[i].p isa AbstractArray ? Array(probs[i].p) : probs[i].p
                    for i in 1:length(I)
            )
            sol,
                solus = batch_solve_up(
                ensembleprob, probs, alg, ensemblealg, I, u0, p;
                adaptive, kwargs...
            )
            [
                ensembleprob.output_func(
                    SciMLBase.build_solution(
                        probs[i], alg, sol.t,
                        solus[i],
                        stats = sol.stats,
                        retcode = sol.retcode
                    ),
                    _make_ensemble_context(I[i], sim_seeds, rng_func, master_rng)
                )[1]
                    for i in 1:length(probs)
            ]
        end
    end
end

function batch_solve_up_kernel(
        ensembleprob, probs, adapted_probs, alg, ensemblealg, I, adaptive;
        kwargs...
    )
    _callback = CallbackSet(generate_callback(_unwrap_kernel_host(probs[1]), length(I), ensemblealg; kwargs...))

    _callback = CallbackSet(
        convert.(
            DiffEqGPU.GPUDiscreteCallback,
            _callback.discrete_callbacks
        )...,
        convert.(
            DiffEqGPU.GPUContinuousCallback,
            _callback.continuous_callbacks
        )...
    )

    dev = ensemblealg.dev
    probs = _kernel_transfer(dev, adapted_probs)

    if adaptive
        ts,
            us = vectorized_asolve(
            probs, ensembleprob.prob, alg;
            kwargs..., callback = _callback
        )
    else
        ts,
            us = vectorized_solve(
            probs, ensembleprob.prob, alg;
            kwargs..., callback = _callback
        )
    end
    solus = _kernel_transfer(CPU(), us)
    solts = _kernel_transfer(CPU(), ts)
    return (solts, solus)
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

    sol = solve(
        prob, _alg; kwargs..., callback = _callback, merge_callbacks = false,
        internalnorm = diffeqgpunorm
    )

    us = Array.(sol.u)
    solus = [[@view(us[i][:, j]) for i in 1:length(us)] for j in 1:length(probs)]
    return (sol, solus)
end

function seed_duals(
        x::Matrix{V}, ::Type{T},
        ::ForwardDiff.Chunk{N} = ForwardDiff.Chunk(
            @view(x[:, 1]),
            typemax(Int64)
        )
    ) where {
        V, T,
        N,
    }
    seeds = ForwardDiff.construct_seeds(ForwardDiff.Partials{N, V})
    return duals = [
        ForwardDiff.Dual{T}(x[i, j], seeds[i])
            for i in 1:size(x, 1), j in 1:size(x, 2)
    ]
end

function extract_dus(us)
    jsize = size(us[1], 1), ForwardDiff.npartials(us[1][1])
    utype = typeof(ForwardDiff.value(us[1][1]))
    return map(1:size(us[1], 2)) do k
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

function ChainRulesCore.rrule(
        ::typeof(batch_solve_up), ensembleprob, probs, alg,
        ensemblealg, I, u0, p; kwargs...
    )
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

    sol = solve(
        prob, _alg; kwargs..., callback = _callback, merge_callbacks = false,
        internalnorm = diffeqgpunorm
    )

    us = Array.(sol.u)
    solus = [
        [ForwardDiff.value.(@view(us[i][:, j])) for i in 1:length(us)]
            for j in 1:length(probs)
    ]

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
        return (ntuple(_ -> NoTangent(), 7)..., Array(VectorOfArray(adj)))
    end
    return (sol, solus), batch_solve_up_adjoint
end

function solve_batch(
        prob, alg, ensemblealg::EnsembleThreads, II, pmap_batch_size,
        ensemble_rng_state; kwargs...
    )
    if length(II) == 1 || Threads.nthreads() == 1
        return SciMLBase.solve_batch(
            prob, alg, EnsembleSerial(), II, pmap_batch_size,
            ensemble_rng_state; kwargs...
        )
    end

    if prob.prob isa SciMLBase.AbstractJumpProblem && length(II) != 1
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
        SciMLBase.solve_batch(
            prob, alg, EnsembleSerial(), I_local, pmap_batch_size,
            ensemble_rng_state; kwargs...
        )
    end
    return SciMLBase.tighten_container_eltype(batch_data)
end

function tmap(f, args...)
    batch_data = Vector{
        Core.Compiler.return_type(f, Tuple{typeof.(getindex.(args, 1))...}),
    }(
        undef,
        length(args[1])
    )
    Threads.@threads for i in 1:length(args[1])
        batch_data[i] = f(getindex.(args, i)...)
    end
    return reduce(vcat, batch_data)
end
