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

@inline initialization_algorithm(initprob, nlsolve_alg) =
    nlsolve_alg === nothing ? SimpleTrustRegion() : nlsolve_alg

struct InitializationResidual{F, P}
    f::F
    p::P
end

@inline (f::InitializationResidual)(u) = f.f(u, f.p)

@inline function initialization_jacobian(prob, u)
    if SciMLBase.has_jac(prob.f)
        return prob.f.jac(u, prob.p)
    end
    return ForwardDiff.jacobian(InitializationResidual(prob.f, prob.p), u)
end

@inline function static_gauss_newton_step(
        J::StaticMatrix{M, N}, residual
    ) where {M, N}
    if M < N
        return transpose(J) * ((J * transpose(J)) \ residual)
    end
    return (transpose(J) * J) \ (transpose(J) * residual)
end

@inline initialization_residual_norm(residual) = sqrt(sum(abs2, residual))

@inline function gpu_rectangular_gauss_newton_solve(prob, abstol; maxiters = 1000)
    u = prob.u0
    residual = prob.f(u, prob.p)
    for _ in 1:maxiters
        if initialization_residual_norm(residual) <= abstol
            return SciMLBase.build_solution(
                prob, SimpleGaussNewton(), u, residual; retcode = ReturnCode.Success
            )
        end
        J = initialization_jacobian(prob, u)
        u -= static_gauss_newton_step(J, residual)
        residual = prob.f(u, prob.p)
    end
    return SciMLBase.build_solution(
        prob, SimpleGaussNewton(), u, residual; retcode = ReturnCode.MaxIters
    )
end

@inline function solve_initialization_problem(prob, alg, abstol, reltol)
    return SciMLBase.solve(prob, alg; abstol, reltol)
end

@inline function solve_initialization_problem(
        prob::SciMLBase.NonlinearLeastSquaresProblem, alg, abstol, reltol
    )
    nunknowns = length(prob.u0)
    nresiduals = prob.f.resid_prototype === nothing ? nunknowns :
        length(prob.f.resid_prototype)
    if nunknowns == nresiduals
        return SciMLBase.solve(prob, alg; abstol, reltol)
    end
    return gpu_rectangular_gauss_newton_solve(prob, abstol)
end

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
    sol = if SciMLBase.is_trivial_initialization(initdata)
        initprob
    else
        alg = initialization_algorithm(initprob, nlsolve_alg)
        nlsol = solve_initialization_problem(initprob, alg, abstol, reltol)
        SciMLBase.successful_retcode(nlsol) || return u0, p, false
        restore_bounded_initialization(nlsol, initprob)
    end

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
