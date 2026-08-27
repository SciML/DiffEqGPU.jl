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

function next_tag!(tags)
    tag = Symbol("##DiffEqGPUInitializationTag#", length(tags) + 1)
    push!(tags, tag)
    return tag
end

tag_numeric(x::Number, tags) = next_tag!(tags)
tag_numeric(x::AbstractArray, tags) = map(Base.Fix2(tag_numeric, tags), x)
tag_numeric(x::Tuple, tags) = map(Base.Fix2(tag_numeric, tags), x)
tag_numeric(x::NamedTuple, tags) = map(Base.Fix2(tag_numeric, tags), x)
function tag_numeric(x, tags)
    values = ntuple(i -> tag_numeric(getfield(x, i), tags), fieldcount(typeof(x)))
    return typeof(x)(values...)
end

function tag_parameters(p::MTKParameters, tags)
    return MTKParameters(
        tag_numeric(p.tunable, tags),
        tag_numeric(p.initials, tags),
        tag_numeric(p.discrete, tags),
        tag_numeric(p.constant, tags),
        p.nonnumeric,
        p.caches
    )
end

function tag_initialization_problem(prob::SciMLBase.NonlinearLeastSquaresProblem, tags)
    return SciMLBase.NonlinearLeastSquaresProblem{SciMLBase.isinplace(prob)}(
        prob.f,
        tag_numeric(prob.u0, tags),
        tag_parameters(prob.p, tags);
        lb = prob.lb,
        ub = prob.ub,
        prob.kwargs...
    )
end

function tag_initialization_problem(
        prob::Union{SciMLBase.NonlinearProblem, SciMLBase.ImmutableNonlinearProblem}, tags
    )
    return SciMLBase.ImmutableNonlinearProblem{SciMLBase.isinplace(prob)}(
        prob.f,
        tag_numeric(prob.u0, tags),
        tag_parameters(prob.p, tags),
        prob.problem_type;
        prob.kwargs...
    )
end

function tag_ode_problem(prob, tags)
    return SciMLBase.ImmutableODEProblem(
        prob.f,
        tag_numeric(prob.u0, tags),
        prob.tspan,
        tag_parameters(prob.p, tags),
        prob.problem_type;
        prob.kwargs...
    )
end

function source_recipe(x::Union{Number, Symbol}, tags)
    index = findfirst(Base.Fix2(isequal, x), tags)
    index === nothing && error(
        "ModelingToolkit initialization maps must copy numeric values from the ODE or initialization problem."
    )
    return InitializationSourceIndex(index)
end
function source_recipe(x::AbstractArray, tags)
    values = map(Base.Fix2(source_recipe, tags), x)
    return SArray{Tuple{size(x)...}}(values)
end
source_recipe(x::Tuple, tags) = map(Base.Fix2(source_recipe, tags), x)
source_recipe(x::NamedTuple, tags) = map(Base.Fix2(source_recipe, tags), x)
function source_recipe(x, tags)
    values = ntuple(i -> source_recipe(getfield(x, i), tags), fieldcount(typeof(x)))
    return InitializationStructureRecipe{typeof(x)}(values)
end

function make_state_map(initprob, map)
    map === nothing && return nothing
    # Evaluate the host-only MTK map on unique symbolic tags, then retain only the
    # resulting device-compatible gather indices.
    tags = Symbol[]
    tagged_initprob = tag_initialization_problem(initprob, tags)
    return InitializationStateMap(source_recipe(map(tagged_initprob), tags))
end

function make_parameter_map(prob, initprob, map)
    map === nothing && return nothing
    # Parameter maps may select from both the ODE problem and nonlinear solution.
    tags = Symbol[]
    tagged_prob = tag_ode_problem(prob, tags)
    tagged_initprob = tag_initialization_problem(initprob, tags)
    p = map(tagged_prob, tagged_initprob)
    p isa MTKParameters || error(
        "ModelingToolkit initialization parameter maps must return `MTKParameters`."
    )
    recipe = InitializationParameterRecipe(
        source_recipe(p.tunable, tags),
        source_recipe(p.initials, tags),
        source_recipe(p.discrete, tags),
        source_recipe(p.constant, tags)
    )
    return InitializationParameterMap(recipe)
end

function DiffEqGPU.make_initialization_maps_compatible(
        prob, initprob, umap, pmap, ::MTKParameters
    )
    return make_state_map(initprob, umap), make_parameter_map(prob, initprob, pmap)
end

end
