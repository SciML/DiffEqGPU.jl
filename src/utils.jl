diffeqgpunorm(u::AbstractArray, t) = sqrt.(sum(abs2, u) ./ length(u))
diffeqgpunorm(u::Union{AbstractFloat, Complex}, t) = abs(u)
function diffeqgpunorm(u::AbstractArray{<:ForwardDiff.Dual}, t)
    return sqrt.(sum(abs2 ∘ ForwardDiff.value, u) ./ length(u))
end
diffeqgpunorm(u::ForwardDiff.Dual, t) = abs(ForwardDiff.value(u))

make_prob_compatible(prob) = prob

function make_prob_compatible(prob::T) where {T <: ODEProblem}
    return convert(ImmutableODEProblem, _maybe_convert_mass_matrix(prob))
end

"""
Convert non-SArray mass matrices (e.g. Diagonal from MTK) to SMatrix for GPU kernel
compatibility. Only needed when u0 is converted to SVector by adapt_structure.
"""
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
