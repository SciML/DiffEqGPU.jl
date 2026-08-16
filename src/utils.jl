diffeqgpunorm(u::AbstractArray, t) = sqrt.(sum(abs2, u) ./ length(u))
diffeqgpunorm(u::Union{AbstractFloat, Complex}, t) = abs(u)
function diffeqgpunorm(u::AbstractArray{<:ForwardDiff.Dual}, t)
    return sqrt.(sum(abs2 ∘ ForwardDiff.value, u) ./ length(u))
end
diffeqgpunorm(u::ForwardDiff.Dual, t) = abs(ForwardDiff.value(u))

"""
    make_prob_compatible(prob)

Prepare a problem for the lower-level `EnsembleGPUKernel` interface.

For an `ODEProblem`, this converts the problem to an immutable representation and adapts
mass-matrix data when needed. Other problem-like values are returned unchanged. The
resulting problem must still satisfy the selected backend's GPU-compatibility requirements.

# Arguments

  - `prob`: an `ODEProblem` or another problem value accepted by the lower-level ensemble
    interface.

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
make_prob_compatible(prob) = prob

function make_prob_compatible(prob::T) where {T <: ODEProblem}
    return convert(ImmutableODEProblem, _maybe_convert_mass_matrix(prob))
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
