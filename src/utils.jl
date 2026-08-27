diffeqgpunorm(u::AbstractArray, t) = sqrt.(sum(abs2, u) ./ length(u))
diffeqgpunorm(u::Union{AbstractFloat, Complex}, t) = abs(u)
function diffeqgpunorm(u::AbstractArray{<:ForwardDiff.Dual}, t)
    return sqrt.(sum(abs2 ∘ ForwardDiff.value, u) ./ length(u))
end
diffeqgpunorm(u::ForwardDiff.Dual, t) = abs(ForwardDiff.value(u))

"""
    make_prob_compatible(prob)

Prepare a problem for the lower-level `EnsembleGPUKernel` interface.

For an `ODEProblem`, this updates any initialization problem on the host, converts the
initialization problem and its maps to immutable static representations for device-side
solving, and adapts parameter and mass-matrix storage. Other problem-like values are
returned unchanged. The resulting problem must still satisfy the selected backend's
GPU-compatibility requirements.

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
make_prob_compatible(prob; kwargs...) = prob
make_parameter_compatible(p) = p

function make_prob_compatible(prob::T) where {T <: ODEProblem}
    if SciMLBase.has_initialization_data(prob.f)
        return _initialized_problem_compatible(prob)
    end

    prob = remake(prob; p = make_parameter_compatible(prob.p))
    return convert(ImmutableODEProblem, _maybe_convert_mass_matrix(prob))
end

function _initialized_problem_compatible(prob)
    oldf = prob.f
    initdata = oldf.initialization_data
    initprob = initdata.initializeprob
    if initdata.update_initializeprob! !== nothing
        initprob = if initdata.is_update_oop === Val(true)
            initdata.update_initializeprob!(initprob, prob)
        else
            initdata.update_initializeprob!(initprob, prob)
            initprob
        end
    end
    initprobmap, initprobpmap = make_initialization_maps_compatible(
        prob, initprob, initdata.initializeprobmap, initdata.initializeprobpmap, prob.p
    )
    compatible_initdata = SciMLBase.OverrideInitData(
        make_nonlinear_problem_compatible(initprob), nothing, initprobmap, initprobpmap,
        nothing, Val(false)
    )

    mass_matrix = _compatible_mass_matrix(oldf.mass_matrix, length(prob.u0))
    newf = SciMLBase.ODEFunction{
        SciMLBase.isinplace(oldf), SciMLBase.specialization(oldf),
    }(
        oldf.f;
        jac = oldf.jac,
        mass_matrix,
        initialization_data = compatible_initdata
    )
    static_u0 = make_static_storage(prob.u0)
    static_p = make_parameter_compatible(prob.p)
    return ImmutableODEProblem(
        newf, static_u0, prob.tspan, static_p, prob.problem_type; prob.kwargs...
    )
end

make_initialization_maps_compatible(prob, initprob, umap, pmap, p) = (umap, pmap)

make_static_storage(x::StaticArrays.StaticArray) =
    StaticArrays.SArray{Tuple{size(x)...}}(map(make_static_storage, x))
make_static_storage(x::Array) =
    StaticArrays.SArray{Tuple{size(x)...}}(map(make_static_storage, x))
make_static_storage(x::Tuple) = map(make_static_storage, x)
make_static_storage(x::NamedTuple) = map(make_static_storage, x)
make_static_storage(x) = x

function make_nonlinear_function_compatible(oldf)
    return SciMLBase.NonlinearFunction{
        false, SciMLBase.FullSpecialize,
    }(
        oldf.f;
        resid_prototype = make_static_storage(oldf.resid_prototype)
    )
end

function make_nonlinear_problem_compatible(prob::SciMLBase.NonlinearLeastSquaresProblem)
    (prob.lb === nothing && prob.ub === nothing) || error(
        "Bounded nonlinear initialization problems are not supported by EnsembleGPUKernel."
    )
    nunknowns = length(prob.u0)
    nresiduals = if prob.f.resid_prototype === nothing
        nunknowns
    else
        length(prob.f.resid_prototype)
    end
    if nresiduals != nunknowns
        status = nresiduals < nunknowns ? "underdetermined" : "overdetermined"
        throw(ArgumentError(
            "$status nonlinear initialization problems are not supported by EnsembleGPUKernel: $nresiduals residuals for $nunknowns unknowns."
        ))
    end

    static_u0 = make_static_storage(prob.u0)
    static_p = make_parameter_compatible(prob.p)
    return SciMLBase.NonlinearLeastSquaresProblem{false}(
        make_nonlinear_function_compatible(prob.f), static_u0, static_p;
        lb = prob.lb,
        ub = prob.ub,
        prob.kwargs...
    )
end

function make_nonlinear_problem_compatible(
        prob::Union{SciMLBase.NonlinearProblem, SciMLBase.ImmutableNonlinearProblem}
    )
    return SciMLBase.ImmutableNonlinearProblem{false}(
        make_nonlinear_function_compatible(prob.f),
        make_static_storage(prob.u0),
        make_parameter_compatible(prob.p),
        prob.problem_type;
        prob.kwargs...
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
