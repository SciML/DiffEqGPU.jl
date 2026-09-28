struct BoundedInitializationFunction{F, L, U}
    f::F
    lb::L
    ub::U
end

"""
Device-compatible representation of an SCC-split initialization problem.

`problem` is the out-of-place residual for the complete initialization system. `blocks`
stores the statically known row/state range of each SCC and whether the block is linear.
Keeping only immutable, fully specialized data avoids the mutable cache writers used by
the host `SCCNonlinearProblem` representation.
"""
struct ImmutableSCCNonlinearProblem{P, B}
    problem::P
    blocks::B
end

struct ImmutableSCCBlock{I, N, L} end

struct ImmutableSCCInitialization{B}
    blocks::B
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

struct SCCBlockResidual{F, P, U, I, N}
    f::F
    p::P
    full_u::U
end

@inline function replace_scc_block(
        full_u::StaticVector{M}, block_u, ::Val{I}, ::Val{N}
    ) where {M, I, N}
    values = ntuple(Val(M)) do j
        I <= j < I + N ? block_u[j - I + 1] : full_u[j]
    end
    return SVector(values)
end

@inline function (f::SCCBlockResidual{F, P, U, I, N})(block_u) where {F, P, U, I, N}
    full_u = replace_scc_block(f.full_u, block_u, Val(I), Val(N))
    residual = f.f(full_u, f.p)
    return SVector{N}(ntuple(i -> residual[I + i - 1], Val(N)))
end
@inline (f::SCCBlockResidual)(block_u, p) = f(block_u)

@inline function scc_block_state(full_u, ::ImmutableSCCBlock{I, N}) where {I, N}
    return SVector{N}(ntuple(i -> full_u[I + i - 1], Val(N)))
end

@inline function scc_block_residual(prob, full_u, ::ImmutableSCCBlock{I, N}) where {I, N}
    return SCCBlockResidual{typeof(prob.f), typeof(prob.p), typeof(full_u), I, N}(
        prob.f, prob.p, full_u
    )
end

@inline replace_scc_block(full_u, block_u, ::ImmutableSCCBlock{I, N}) where {I, N} =
    replace_scc_block(full_u, block_u, Val(I), Val(N))

# The Jacobian is re-derived here instead of reusing the `A`/`b` of MTK's linear SCC block:
# those come from a mutable update function of the upstream blocks' solutions, so they
# would have to be re-evaluated per trajectory on the device anyway. Differentiating the
# flattened residual yields the same matrix while keeping the update wrappers on the host.
@inline function solve_scc_block(
        prob, full_u, block::ImmutableSCCBlock{I, N, true}, alg, abstol, reltol
    ) where {I, N}
    residual = scc_block_residual(prob, full_u, block)
    u0 = scc_block_state(full_u, block)
    r0 = residual(u0)
    # A linear block's residual is affine, so one Newton step is exact. Solving for the step
    # avoids forming `J * u0`, which cancels against `r0` for states large next to the step.
    u = u0 - linear_solve(ForwardDiff.jacobian(residual, u0), r0)
    tolerance = abstol + reltol * initialization_residual_norm(r0)
    return u, initialization_residual_norm(residual(u)) <= tolerance
end

@inline function solve_scc_block(
        prob, full_u, block::ImmutableSCCBlock{I, N, false}, alg, abstol, reltol
    ) where {I, N}
    block_prob = SciMLBase.ImmutableNonlinearProblem{false}(
        scc_block_residual(prob, full_u, block), scc_block_state(full_u, block)
    )
    sol = SciMLBase.solve(block_prob, alg; abstol, reltol)
    return sol.u, SciMLBase.successful_retcode(sol)
end

@inline solve_scc_blocks(prob, full_u, ::Tuple{}, alg, abstol, reltol) = (full_u, true)
@inline function solve_scc_blocks(prob, full_u, blocks::Tuple, alg, abstol, reltol)
    block = first(blocks)
    block_u, success = solve_scc_block(prob, full_u, block, alg, abstol, reltol)
    success || return full_u, false
    updated_u = replace_scc_block(full_u, block_u, block)
    return solve_scc_blocks(prob, updated_u, Base.tail(blocks), alg, abstol, reltol)
end

@inline function solve_initialization_problem(
        nonlinear_prob, metadata::ImmutableSCCInitialization, alg, abstol, reltol
    )
    u, success = solve_scc_blocks(
        nonlinear_prob, nonlinear_prob.u0, metadata.blocks, alg, abstol, reltol
    )
    residual = nonlinear_prob.f(u, nonlinear_prob.p)
    # Block-level success trusts the lowered block layout; the full residual also has to
    # vanish, so a layout that is not actually block-triangular cannot pass silently.
    r0 = nonlinear_prob.f(nonlinear_prob.u0, nonlinear_prob.p)
    tolerance = abstol + reltol * initialization_residual_norm(r0)
    success = success && initialization_residual_norm(residual) <= tolerance
    retcode = success ? ReturnCode.Success : ReturnCode.Failure
    return SciMLBase.build_solution(nonlinear_prob, alg, u, residual; retcode)
end

@inline solve_initialization_problem(prob, metadata, alg, abstol, reltol) =
    solve_initialization_problem(prob, alg, abstol, reltol)

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
        nlsol = solve_initialization_problem(
            initprob, initdata.metadata, alg, abstol, reltol
        )
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
