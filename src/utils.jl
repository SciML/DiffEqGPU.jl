diffeqgpunorm(u::AbstractArray, t) = sqrt.(sum(abs2, u) ./ length(u))
diffeqgpunorm(u::Union{AbstractFloat, Complex}, t) = abs(u)
function diffeqgpunorm(u::AbstractArray{<:ForwardDiff.Dual}, t)
    return sqrt.(sum(abs2 ∘ ForwardDiff.value, u) ./ length(u))
end
diffeqgpunorm(u::ForwardDiff.Dual, t) = abs(ForwardDiff.value(u))

"""
    make_prob_compatible(prob; nlsolve_alg = SimpleTrustRegion(), abstol = 1e-6,
        reltol = 1e-6)

Prepare a problem for the lower-level `EnsembleGPUKernel` interface.

For an `ODEProblem`, this solves any initialization problem on the host, removes the
initialization metadata, converts the problem to an immutable representation, and adapts
parameter and mass-matrix storage when needed. Other problem-like values are returned
unchanged. The resulting problem must still satisfy the selected backend's
GPU-compatibility requirements.

# Arguments

  - `prob`: an `ODEProblem` or another problem value accepted by the lower-level ensemble
    interface.
  - `nlsolve_alg`: nonlinear solver used for problem initialization.
  - `abstol`: absolute tolerance used for problem initialization.
  - `reltol`: relative tolerance used for problem initialization.

# Returns

An immutable, backend-compatible representation for ODE problems, or `prob` unchanged for
other values.

# Examples

```julia
using DiffEqGPU, SciMLBase, StaticArrays
f(u, p, t) = u
prob = ODEProblem{false}(f, SVector(1.0f0), (0.0f0, 1.0f0), SVector(1.0f0))
gpu_prob = DiffEqGPU.make_prob_compatible(prob)
```
"""
make_prob_compatible(prob; kwargs...) = prob
make_parameter_compatible(p) = p

function make_prob_compatible(
        prob::T; nlsolve_alg = SimpleTrustRegion(), abstol = 1.0e-6,
        reltol = 1.0e-6
    ) where {T <: ODEProblem}
    if SciMLBase.has_initialization_data(prob.f)
        return _initialized_problem_compatible(prob, nlsolve_alg, abstol, reltol)
    end

    prob = remake(prob; p = make_parameter_compatible(prob.p))
    return convert(ImmutableODEProblem, _maybe_convert_mass_matrix(prob))
end

function _initialized_problem_compatible(prob, nlsolve_alg, abstol, reltol)
    u0, p, success = gpu_initialization_solve(prob, nlsolve_alg, abstol, reltol)
    success || error("Initialization failed while preparing an ODEProblem for EnsembleGPUKernel.")

    oldf = prob.f
    mass_matrix = _compatible_mass_matrix(oldf.mass_matrix, length(u0))
    newf = SciMLBase.ODEFunction{
        SciMLBase.isinplace(oldf), SciMLBase.specialization(oldf),
    }(
        oldf.f;
        jac = oldf.jac,
        mass_matrix,
        initialization_data = nothing
    )
    static_u0 = StaticArrays.SVector{length(u0)}(u0)
    static_p = make_parameter_compatible(p)
    return ImmutableODEProblem(
        newf, static_u0, prob.tspan, static_p, prob.problem_type; prob.kwargs...
    )
end

function _compatible_mass_matrix(mm, N)
    (mm isa StaticArrays.StaticArray || mm === LinearAlgebra.I) && return mm
    return StaticArrays.SMatrix{N, N}(mm)
end

function _maybe_convert_mass_matrix(prob)
    mm = prob.f.mass_matrix
    # Already an SArray, UniformScaling, or I — nothing to do
    (mm isa StaticArrays.StaticArray || mm === LinearAlgebra.I) && return prob
    # Convert to SMatrix
    N = length(prob.u0)
    smm = StaticArrays.SMatrix{N, N}(mm)
    oldf = prob.f
    newf = SciMLBase.ODEFunction{SciMLBase.isinplace(oldf), SciMLBase.specialization(oldf)}(
        oldf.f;
        jac = oldf.jac,
        mass_matrix = smm,
        initialization_data = oldf.initialization_data
    )
    return remake(prob; f = newf)
end
