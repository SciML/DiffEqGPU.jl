module ModelingToolkitBaseExt

using ModelingToolkitBase: MTKParameters, System, unknowns
using Symbolics
using RuntimeGeneratedFunctions: drop_expr
using StaticArraysCore: SArray, StaticArray, SVector
import DiffEqGPU
import SciMLBase

function static_parameter_storage(x::StaticArray)
    values = map(static_parameter_storage, x)
    return SArray{Tuple{size(x)...}}(values)
end
function static_parameter_storage(x::Array)
    values = map(static_parameter_storage, x)
    return SArray{Tuple{size(x)...}}(values)
end
static_parameter_storage(x::Tuple) = map(static_parameter_storage, x)
static_parameter_storage(x::NamedTuple) = map(static_parameter_storage, x)
static_parameter_storage(x) = x

function DiffEqGPU.make_parameter_compatible(p::MTKParameters)
    return MTKParameters(
        static_parameter_storage(p.tunable),
        static_parameter_storage(p.initials),
        static_parameter_storage(p.discrete),
        static_parameter_storage(p.constant),
        static_parameter_storage(p.nonnumeric),
        static_parameter_storage(p.caches)
    )
end

function DiffEqGPU.lower_initialization_problem(prob::SciMLBase.SCCNonlinearProblem)
    sys = prob.f.sys
    sys isa System || throw(
        ArgumentError(
            "Only ModelingToolkit-generated SCC nonlinear initialization problems can be lowered for EnsembleGPUKernel."
        )
    )
    any(p -> nameof(typeof(p)) === :HomotopyProblem, prob.probs) && throw(
        ArgumentError(
            "SCC nonlinear initialization problems containing homotopy blocks are not supported by EnsembleGPUKernel."
        )
    )

    block_states = map(prob.probs) do block_prob
        block_u0 = SciMLBase.state_values(block_prob)
        block_u0 !== nothing && return block_u0
        # A linear block is solved by one exact Newton step, which lands on the solution
        # from any seed, so a zero seed stands in for the missing state.
        block_prob isa SciMLBase.LinearProblem || throw(
            ArgumentError(
                "Every nonlinear SCC initialization block must have an initial state for EnsembleGPUKernel."
            )
        )
        zero(block_prob.b)
    end
    u0 = reduce(vcat, block_states)
    length(u0) == length(unknowns(sys)) || throw(
        ArgumentError("SCC initialization block sizes do not match the full state size.")
    )
    f = SciMLBase.NonlinearFunction{false, SciMLBase.FullSpecialize}(
        sys; u0, p = prob.p, check_compatibility = false
    )
    nonlinear_prob = SciMLBase.NonlinearProblem{false}(f, u0, prob.p)

    offset = 0
    blocks = map(prob.probs, block_states) do block_prob, block_u0
        n = length(block_u0)
        block = DiffEqGPU.ImmutableSCCBlock{
            offset + 1, n, block_prob isa SciMLBase.LinearProblem,
        }()
        offset += n
        block
    end
    return DiffEqGPU.ImmutableSCCNonlinearProblem(nonlinear_prob, Tuple(blocks))
end

struct InitializationSourceIndex{I} end

struct InitializationArrayRecipe{A, R}
    values::R
end

InitializationArrayRecipe{A}(values::R) where {A, R} =
    InitializationArrayRecipe{A, R}(values)

struct InitializationStructureRecipe{T, R}
    values::R
end

struct InitializationConstant{V}
    value::V
end

struct InitializationScalarExpression{F}
    f::F
end

struct InitializationArrayExpression{A, F}
    f::F
end

InitializationArrayExpression{A}(f::F) where {A, F} = InitializationArrayExpression{A, F}(f)

struct InitializationStateMap{R}
    recipe::R
end

struct InitializationParameterRecipe{T, I, D, C}
    tunable::T
    initials::I
    discrete::D
    constant::C
end

struct InitializationParameterMap{R}
    recipe::R
end

function numeric_values(x::Number, ::Type{T}) where {T}
    return SVector{1, T}(x)
end
numeric_values(x::StaticArray, ::Type{T}) where {T} = numeric_values(Tuple(x), T)
numeric_values(x::NamedTuple, ::Type{T}) where {T} = numeric_values(values(x), T)
numeric_values(::Tuple{}, ::Type{T}) where {T} = SVector{0, T}()
function numeric_values(x::Tuple, ::Type{T}) where {T}
    return vcat(numeric_values(first(x), T), numeric_values(Base.tail(x), T))
end
function numeric_values(x, ::Type{T}) where {T}
    values = ntuple(i -> getfield(x, i), fieldcount(typeof(x)))
    return numeric_values(values, T)
end

function initialization_sources(valp)
    u = SciMLBase.state_values(valp)
    p = SciMLBase.parameter_values(valp)
    T = u === nothing ? eltype(p.tunable) : eltype(u)
    return vcat(
        numeric_values(u, T), numeric_values(p.tunable, T),
        numeric_values(p.initials, T), numeric_values(p.discrete, T),
        numeric_values(p.constant, T)
    )
end

gather_initialization_value(::InitializationSourceIndex{I}, sources) where {I} = sources[I]
gather_initialization_value(recipe::InitializationConstant, sources) = recipe.value
gather_initialization_value(recipe::InitializationScalarExpression, sources) =
    recipe.f(sources)
function gather_initialization_value(
        recipe::InitializationArrayExpression{A}, sources
    ) where {A}
    return A(Tuple(recipe.f(sources)))
end
function gather_initialization_value(
        recipe::InitializationArrayRecipe{A}, sources
    ) where {A}
    values = map(Base.Fix2(gather_initialization_value, sources), recipe.values)
    return A(values)
end
gather_initialization_value(recipe::StaticArray, sources) =
    map(Base.Fix2(gather_initialization_value, sources), recipe)
gather_initialization_value(recipe::Tuple, sources) =
    map(Base.Fix2(gather_initialization_value, sources), recipe)
gather_initialization_value(recipe::NamedTuple, sources) =
    map(Base.Fix2(gather_initialization_value, sources), recipe)
function gather_initialization_value(
        recipe::InitializationStructureRecipe{T}, sources
    ) where {T}
    values = map(Base.Fix2(gather_initialization_value, sources), recipe.values)
    return T(values...)
end

function (map::InitializationStateMap)(sol)
    return gather_initialization_value(map.recipe, initialization_sources(sol))
end

function (map::InitializationParameterMap)(prob, sol)
    sources = vcat(initialization_sources(prob), initialization_sources(sol))
    recipe = map.recipe
    p = SciMLBase.parameter_values(prob)
    return MTKParameters(
        gather_initialization_value(recipe.tunable, sources),
        gather_initialization_value(recipe.initials, sources),
        gather_initialization_value(recipe.discrete, sources),
        gather_initialization_value(recipe.constant, sources),
        p.nonnumeric,
        p.caches
    )
end

struct InitializationSourceTrace
    vars::Vector{Symbolics.Num}
    lookup::Dict{Any, Int}
end
InitializationSourceTrace() = InitializationSourceTrace(Symbolics.Num[], Dict{Any, Int}())

function next_source_token!(trace::InitializationSourceTrace)
    i = length(trace.vars) + 1
    var = Symbolics.variable(:ˍdiffeqgpu_source, i)
    push!(trace.vars, var)
    trace.lookup[Symbolics.unwrap(var)] = i
    return var
end

source_index_of(trace::InitializationSourceTrace, x) =
    get(trace.lookup, Symbolics.unwrap(x), nothing)

issymbolic(x) = Symbolics.unwrap(x) isa Symbolics.BasicSymbolic

index_numeric(::Number, trace) = next_source_token!(trace)
index_numeric(x::AbstractArray, trace) = map(Base.Fix2(index_numeric, trace), x)
index_numeric(x::Tuple, trace) = map(Base.Fix2(index_numeric, trace), x)
index_numeric(x::NamedTuple, trace) = map(Base.Fix2(index_numeric, trace), x)
function index_numeric(x, trace)
    values = ntuple(i -> index_numeric(getfield(x, i), trace), fieldcount(typeof(x)))
    return typeof(x)(values...)
end

function index_parameters(p::MTKParameters, trace)
    return MTKParameters(
        index_numeric(p.tunable, trace),
        index_numeric(p.initials, trace),
        index_numeric(p.discrete, trace),
        index_numeric(p.constant, trace),
        p.nonnumeric,
        p.caches
    )
end

function index_initialization_problem(prob::SciMLBase.NonlinearLeastSquaresProblem, trace)
    return SciMLBase.NonlinearLeastSquaresProblem{SciMLBase.isinplace(prob)}(
        prob.f,
        index_numeric(prob.u0, trace),
        index_parameters(prob.p, trace);
        lb = prob.lb,
        ub = prob.ub,
        prob.kwargs...
    )
end

function index_initialization_problem(
        prob::DiffEqGPU.ImmutableSCCNonlinearProblem, trace
    )
    return index_initialization_problem(prob.problem, trace)
end

function index_initialization_problem(
        prob::Union{SciMLBase.NonlinearProblem, SciMLBase.ImmutableNonlinearProblem}, trace
    )
    return SciMLBase.ImmutableNonlinearProblem{SciMLBase.isinplace(prob)}(
        prob.f,
        index_numeric(prob.u0, trace),
        index_parameters(prob.p, trace),
        prob.problem_type;
        prob.kwargs...
    )
end

function index_ode_problem(prob, trace)
    return SciMLBase.ImmutableODEProblem(
        prob.f,
        index_numeric(prob.u0, trace),
        prob.tspan,
        index_parameters(prob.p, trace),
        prob.problem_type;
        prob.kwargs...
    )
end

@inline function compile_sources_function(exprs, trace::InitializationSourceTrace)
    built = Symbolics.build_function(exprs, trace.vars; expression = Val(false))
    f = built isa Tuple ? built[1] : built
    # An RGF's stored `body::Expr` is only for re-generation; dropping it makes the
    # callable isbits so the recipe can live inside GPU kernel problems.
    return drop_expr(f)
end

function scalar_source_recipe(x, trace::InitializationSourceTrace)
    if issymbolic(x)
        i = source_index_of(trace, x)
        i === nothing || return InitializationSourceIndex{i}()
        return InitializationScalarExpression(
            compile_sources_function(Symbolics.unwrap(x), trace)
        )
    end
    x isa Number && return InitializationConstant(x)
    return error(
        "ModelingToolkit initialization maps produced an unsupported value of type $(typeof(x))."
    )
end

# A map output entry is either a value copied verbatim from a source slot (a bare traced
# variable), a literal constant, or a computed symbolic expression. Whole arrays with any
# computed entry are compiled into one generated gather-and-compute function so the device
# recipe stays a single call.
is_direct_entry(x, trace) = !issymbolic(x) || source_index_of(trace, x) !== nothing

function source_recipe(x, trace::InitializationSourceTrace)
    (x isa Number || issymbolic(x)) && return scalar_source_recipe(x, trace)
    values = ntuple(
        i -> source_recipe(getfield(x, i), trace), fieldcount(typeof(x))
    )
    return InitializationStructureRecipe{typeof(x)}(values)
end
function source_recipe(x::AbstractArray, trace::InitializationSourceTrace)
    storage_type = SArray{Tuple{size(x)...}}
    return array_source_recipe(x, trace, storage_type)
end
source_recipe(x::Tuple, trace::InitializationSourceTrace) =
    map(Base.Fix2(source_recipe, trace), x)
source_recipe(x::NamedTuple, trace::InitializationSourceTrace) =
    map(Base.Fix2(source_recipe, trace), x)
function source_recipe(
        x::Union{Symbolics.Num, Symbolics.BasicSymbolic}, trace::InitializationSourceTrace
    )
    return scalar_source_recipe(x, trace)
end

function array_source_recipe(x, trace, storage_type)
    if all(el -> is_direct_entry(el, trace), x)
        values = ntuple(i -> scalar_source_recipe(x[i], trace), length(x))
        return InitializationArrayRecipe{storage_type}(values)
    end
    exprs = SVector{length(x)}(map(Symbolics.unwrap, vec(x))...)
    return InitializationArrayExpression{storage_type}(
        compile_sources_function(exprs, trace)
    )
end

source_recipe(x, trace::InitializationSourceTrace, prototype) = source_recipe(x, trace)
function source_recipe(
        x::AbstractArray, trace::InitializationSourceTrace, prototype::AbstractArray
    )
    storage_type = typeof(static_parameter_storage(prototype))
    return array_source_recipe(x, trace, storage_type)
end
function source_recipe(x::Tuple, trace::InitializationSourceTrace, prototype::Tuple)
    return map(
        (value, target) -> source_recipe(value, trace, target), x, prototype
    )
end
function source_recipe(
        x::NamedTuple, trace::InitializationSourceTrace, prototype::NamedTuple
    )
    values = map(
        (value, target) -> source_recipe(value, trace, target), x, prototype
    )
    return NamedTuple{keys(x)}(values)
end

function make_state_map(initprob, map)
    map === nothing && return nothing
    # Evaluate the host-only MTK map on symbolically traced sources: copied slots become
    # static gather recipes and computed entries compile into generated device functions.
    trace = InitializationSourceTrace()
    indexed_initprob = index_initialization_problem(initprob, trace)
    return InitializationStateMap(source_recipe(map(indexed_initprob), trace))
end

function make_parameter_map(prob, initprob, map)
    map === nothing && return nothing
    # Parameter maps may select from both the ODE problem and nonlinear solution.
    trace = InitializationSourceTrace()
    indexed_prob = index_ode_problem(prob, trace)
    indexed_initprob = index_initialization_problem(initprob, trace)
    p = map(indexed_prob, indexed_initprob)
    p isa MTKParameters || error(
        "ModelingToolkit initialization parameter maps must return `MTKParameters`."
    )
    prototype = SciMLBase.parameter_values(prob)
    recipe = InitializationParameterRecipe(
        source_recipe(p.tunable, trace, prototype.tunable),
        source_recipe(p.initials, trace, prototype.initials),
        source_recipe(p.discrete, trace, prototype.discrete),
        source_recipe(p.constant, trace, prototype.constant)
    )
    return InitializationParameterMap(recipe)
end

function DiffEqGPU.make_initialization_maps_compatible(
        prob, initprob, umap, pmap, ::MTKParameters
    )
    return make_state_map(initprob, umap), make_parameter_map(prob, initprob, pmap)
end

end
