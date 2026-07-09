"""
    AbstractNLSolver

Developer interface for nonlinear solver state used by stiff `EnsembleGPUKernel`
integrators.

# Interface Rules

Subtypes are mutable-by-replacement state containers used inside GPU kernel integrators.
They must store the current stage correction, temporary stage state, derivative scaling,
Jacobian/W-operator constructors, time-step metadata, parameters, and iteration counters
needed by `nlsolve`. A subtype must be usable from GPU-compiled code: all fields and all
functions called from `nlsolve` must be concrete and GPU compatible, and the state update
must return the updated solver value rather than relying on host mutation.

Implicit kernel algorithms build this state with `build_nlsolver` and then call
`nlsolve(nlsolver, integrator)` from the generic step implementation. The required public
behavior is tested through generic stiff `EnsembleGPUKernel` solves and the lower-level
`vectorized_solve`/`vectorized_asolve` paths rather than by inspecting solver fields.

# Fields

Concrete subtypes are expected to provide the fields read by `nlsolve`, including `z`,
`tmp`, `γ`, `c`, `J`, `W`, `dt`, `t`, `p`, `iter`, and `maxiters`.
"""
abstract type AbstractNLSolver end

"""
    AbstractNLSolverCache

Developer interface for nonlinear solver cache objects used by future
`EnsembleGPUKernel` nonlinear solver implementations.

# Interface Rules

Subtypes are reserved for GPU-compatible nonlinear solver caches. Cache fields must be
static or otherwise acceptable to KernelAbstractions kernels, and methods using the cache
must be callable from device code. A cache must not depend on host-only allocation,
reflection, dynamic dispatch, BLAS/LAPACK calls, or non-bitstype closures in the kernel
step path.

This is developer-facing API for DiffEqGPU solver implementations. User code should select
documented algorithms such as `GPURodas4` or `GPUKvaerno5` instead of constructing
nonlinear solver caches directly.
"""
abstract type AbstractNLSolverCache end

"""
    NLSolver{uType, gamType, tmpType, tType, JType, WType, pType} <: AbstractNLSolver

Concrete Newton-style nonlinear solver state used by stiff `EnsembleGPUKernel`
integrators.

# Fields

  - `z`: current nonlinear correction.
  - `tmp`: stage state used by DIRK and multistep methods.
  - `tmp2`: additional temporary state for methods that need a second work vector.
  - `ztmp`: temporary correction storage.
  - `γ`: method coefficient multiplying the nonlinear correction.
  - `c`: stage abscissa.
  - `α`: method-specific stage coefficient.
  - `κ`: nonlinear convergence damping parameter.
  - `J`: callable Jacobian builder `J(u, p, t)`.
  - `W`: callable W-operator builder `W(u, p, t)`.
  - `dt`: current step size.
  - `t`: current step start time.
  - `p`: parameters for the current trajectory.
  - `iter`: current nonlinear iteration count.
  - `maxiters`: maximum nonlinear iterations.

# Interface Rules

`NLSolver` is constructed by `build_nlsolver`, consumed by `nlsolve`, and updated by
returning a new value with modified fields. It is not intended as a direct user
constructor; users select a stiff GPU algorithm and provide GPU-compatible derivative
functions or enable algorithm-controlled differentiation.
"""
struct NLSolver{uType, gamType, tmpType, tType, JType, WType, pType} <: AbstractNLSolver
    z::uType
    tmp::uType # DIRK and multistep methods only use tmp
    tmp2::tmpType # for GLM if neccssary
    ztmp::uType
    γ::gamType
    c::tType
    α::tType
    κ::tType
    J::JType
    W::WType
    dt::tType
    t::tType
    p::pType
    iter::Int
    maxiters::Int
end

function NLSolver{tType}(
        z, tmp, ztmp, γ, c, α, κ, J, W, dt, t, p,
        iter, maxiters, tmp2 = nothing
    ) where {tType}
    return NLSolver{typeof(z), typeof(γ), typeof(tmp2), tType, typeof(J), typeof(W), typeof(p)}(
        z,
        tmp,
        tmp2,
        ztmp,
        γ,
        convert(
            tType,
            c
        ),
        convert(
            tType,
            α
        ),
        convert(
            tType,
            κ
        ),
        J,
        W,
        dt,
        t,
        p,
        iter,
        maxiters
    )
end

@inline function build_J_W(alg, f, γ, dt)
    J(u, p, t) =
    if SciMLBase.has_jac(f)
        f.jac(u, p, t)
    elseif alg_autodiff(alg)
        ForwardDiff.jacobian(u -> f(u, p, t), u)
    else
        finite_diff_jac(u -> f(u, p, t), f.jac_prototype, u)
    end
    W(u, p, t) = -f.mass_matrix + γ * dt * J(u, p, t)
    return J, W
end

@inline function build_tgrad(alg, f)
    function tgrad(u, p, t)
        return if SciMLBase.has_tgrad(f)
            f.tgrad(u, p, t)
        elseif alg_autodiff(alg)
            ForwardDiff.derivative(t -> f(u, p, t), t)
        else
            # derivative using finite difference
            begin
                dt = sqrt(eps(eltype(t)))
                (f(u, p, t + dt) - f(u, p, t)) / dt
            end
        end
    end
    return tgrad
end

@inline function build_nlsolver(
        alg, u, p,
        t, dt,
        f,
        γ, c
    )
    return build_nlsolver(alg, u, p, t, dt, f, γ, c, 1)
end

@inline function build_nlsolver(
        alg, u, p,
        t, dt,
        f,
        γ, c, α
    )
    # define fields of non-linear solver
    z = u
    tmp = u
    ztmp = u
    J, W = build_J_W(alg, f, γ, dt)
    max_iter = 30
    κ = 1 / 100
    return NLSolver{typeof(dt)}(
        z, tmp, ztmp, γ, c, α, κ,
        J, W, dt, t, p, 0, max_iter
    )
end
