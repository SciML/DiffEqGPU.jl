struct BoundedInitializationFunction{F, L, U}
    f::F
    lb::L
    ub::U
end

@inline function _from_unbounded_initialization(u, lb, ub)
    if isfinite(lb) && isfinite(ub)
        return lb + (ub - lb) / (one(u) + exp(-u))
    elseif isfinite(lb)
        return lb + exp(u)
    elseif isfinite(ub)
        return ub - exp(u)
    else
        return u
    end
end

@inline function (f::BoundedInitializationFunction)(u, p)
    bounded_u = map(_from_unbounded_initialization, u, f.lb, f.ub)
    return f.f(bounded_u, p)
end

@inline initialization_algorithm(
    ::SciMLBase.NonlinearLeastSquaresProblem, nlsolve_alg
) = SimpleGaussNewton()
@inline initialization_algorithm(initprob, nlsolve_alg) =
    nlsolve_alg === nothing ? SimpleTrustRegion() : nlsolve_alg

@inline restore_bounded_initialization(sol, initprob) = sol
@inline function restore_bounded_initialization(
        sol, initprob::SciMLBase.NonlinearLeastSquaresProblem
    )
    f = initprob.f.f
    f isa BoundedInitializationFunction || return sol
    bounded_u = map(_from_unbounded_initialization, sol.u, f.lb, f.ub)
    return @set sol.u = bounded_u
end

@inline function gpu_initialization_solve(prob, nlsolve_alg, abstol, reltol)
    f = prob.f
    u0 = prob.u0
    p = prob.p

    # Check if initialization is actually needed
    if !SciMLBase.has_initialization_data(f) || f.initialization_data === nothing
        return u0, p, true
    end

    initdata = f.initialization_data
    if initdata.initializeprob === nothing
        return u0, p, true
    end

    # Create initialization problem
    initprob = initdata.initializeprob

    # Update the problem if needed — pass full prob so MTK can find the SciMLFunction
    if initdata.update_initializeprob! !== nothing
        if initdata.is_update_oop === Val(true)
            initprob = initdata.update_initializeprob!(initprob, prob)
        else
            initdata.update_initializeprob!(initprob, prob)
        end
    end
    alg = initialization_algorithm(initprob, nlsolve_alg)

    # Solve initialization problem using SimpleNonlinearSolve
    sol = SciMLBase.solve(initprob, alg; abstol, reltol)

    # Extract results — initialization must succeed
    if !SciMLBase.successful_retcode(sol)
        return u0, p, false
    end

    sol = restore_bounded_initialization(sol, initprob)

    # Apply result mappings if they exist, converting back to the original u0 type
    u_init = if initdata.initializeprobmap !== nothing
        raw = initdata.initializeprobmap(sol)
        typeof(u0)(raw)
    else
        u0
    end

    p_init = if initdata.initializeprobpmap !== nothing
        initdata.initializeprobpmap(prob, sol)
    else
        p
    end

    return u_init, p_init, true
end
