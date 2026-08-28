module ModelingToolkitBaseExt

using ModelingToolkitBase: MTKParameters
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

struct InitializationSourceIndex
    index::Int
end

struct InitializationStructureRecipe{T, R}
    values::R
end

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
    T = eltype(u)
    return vcat(
        numeric_values(u, T), numeric_values(p.tunable, T),
        numeric_values(p.initials, T), numeric_values(p.discrete, T),
        numeric_values(p.constant, T)
    )
end

gather_initialization_value(index::InitializationSourceIndex, sources) =
    sources[index.index]
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

function next_source_index!(counter)
    counter[] += 1
    return counter[]
end

index_numeric(::Number, counter) = next_source_index!(counter)
index_numeric(x::AbstractArray, counter) = map(Base.Fix2(index_numeric, counter), x)
index_numeric(x::Tuple, counter) = map(Base.Fix2(index_numeric, counter), x)
index_numeric(x::NamedTuple, counter) = map(Base.Fix2(index_numeric, counter), x)
function index_numeric(x, counter)
    values = ntuple(i -> index_numeric(getfield(x, i), counter), fieldcount(typeof(x)))
    return typeof(x)(values...)
end

function index_parameters(p::MTKParameters, counter)
    return MTKParameters(
        index_numeric(p.tunable, counter),
        index_numeric(p.initials, counter),
        index_numeric(p.discrete, counter),
        index_numeric(p.constant, counter),
        p.nonnumeric,
        p.caches
    )
end

function index_initialization_problem(prob::SciMLBase.NonlinearLeastSquaresProblem, counter)
    return SciMLBase.NonlinearLeastSquaresProblem{SciMLBase.isinplace(prob)}(
        prob.f,
        index_numeric(prob.u0, counter),
        index_parameters(prob.p, counter);
        lb = prob.lb,
        ub = prob.ub,
        prob.kwargs...
    )
end

function index_initialization_problem(
        prob::Union{SciMLBase.NonlinearProblem, SciMLBase.ImmutableNonlinearProblem}, counter
    )
    return SciMLBase.ImmutableNonlinearProblem{SciMLBase.isinplace(prob)}(
        prob.f,
        index_numeric(prob.u0, counter),
        index_parameters(prob.p, counter),
        prob.problem_type;
        prob.kwargs...
    )
end

function index_ode_problem(prob, counter)
    return SciMLBase.ImmutableODEProblem(
        prob.f,
        index_numeric(prob.u0, counter),
        prob.tspan,
        index_parameters(prob.p, counter),
        prob.problem_type;
        prob.kwargs...
    )
end

function source_recipe(index::Number, source_count)
    source_index = try
        Int(index)
    catch
        nothing
    end
    if source_index === nothing || !isequal(index, source_index) ||
            !(1 <= source_index <= source_count)
        error(
            "ModelingToolkit initialization maps must copy numeric values from the ODE or initialization problem."
        )
    end
    return InitializationSourceIndex(source_index)
end
function source_recipe(x::AbstractArray, source_count)
    values = map(Base.Fix2(source_recipe, source_count), x)
    return SArray{Tuple{size(x)...}}(values)
end
source_recipe(x::Tuple, source_count) = map(Base.Fix2(source_recipe, source_count), x)
source_recipe(x::NamedTuple, source_count) = map(Base.Fix2(source_recipe, source_count), x)
function source_recipe(x, source_count)
    values = ntuple(
        i -> source_recipe(getfield(x, i), source_count), fieldcount(typeof(x))
    )
    return InitializationStructureRecipe{typeof(x)}(values)
end

function make_state_map(initprob, map)
    map === nothing && return nothing
    # Evaluate the host-only MTK map on sequential source indices, then retain its
    # device-compatible gather recipe.
    counter = Ref(0)
    indexed_initprob = index_initialization_problem(initprob, counter)
    return InitializationStateMap(source_recipe(map(indexed_initprob), counter[]))
end

function make_parameter_map(prob, initprob, map)
    map === nothing && return nothing
    # Parameter maps may select from both the ODE problem and nonlinear solution.
    counter = Ref(0)
    indexed_prob = index_ode_problem(prob, counter)
    indexed_initprob = index_initialization_problem(initprob, counter)
    p = map(indexed_prob, indexed_initprob)
    p isa MTKParameters || error(
        "ModelingToolkit initialization parameter maps must return `MTKParameters`."
    )
    recipe = InitializationParameterRecipe(
        source_recipe(p.tunable, counter[]),
        source_recipe(p.initials, counter[]),
        source_recipe(p.discrete, counter[]),
        source_recipe(p.constant, counter[])
    )
    return InitializationParameterMap(recipe)
end

function DiffEqGPU.make_initialization_maps_compatible(
        prob, initprob, umap, pmap, ::MTKParameters
    )
    return make_state_map(initprob, umap), make_parameter_map(prob, initprob, pmap)
end

end
