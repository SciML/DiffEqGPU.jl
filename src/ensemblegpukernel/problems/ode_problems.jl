import SciMLBase: @add_kwonly, AbstractODEProblem, AbstractODEFunction,
    FunctionWrapperSpecialize, StandardODEProblem, prepare_initial_state, promote_tspan,
    warn_paramtype

struct ImmutableODEProblem{uType, tType, isinplace, P, F, K, PT} <:
       AbstractODEProblem{uType, tType, isinplace}
    """The ODE is `du = f(u,p,t)` for out-of-place and f(du,u,p,t) for in-place."""
    f::F
    """The initial condition is `u(tspan[1]) = u0`."""
    u0::uType
    """The solution `u(t)` will be computed for `tspan[1] ≤ t ≤ tspan[2]`."""
    tspan::tType
    """Constant parameters to be supplied as the second argument of `f`."""
    p::P
    """A callback to be applied to every solver which uses the problem."""
    kwargs::K
    """An internal argument for storing traits about the solving process."""
    problem_type::PT
    @add_kwonly function ImmutableODEProblem{iip}(f::AbstractODEFunction{iip},
        u0, tspan, p = NullParameters(),
        problem_type = StandardODEProblem();
        kwargs...) where {iip}
        _u0 = prepare_initial_state(u0)
        _tspan = promote_tspan(tspan)
        warn_paramtype(p)
        new{typeof(_u0), typeof(_tspan),
            isinplace(f), typeof(p), typeof(f),
            typeof(kwargs),
            typeof(problem_type)}(f,
            _u0,
            _tspan,
            p,
            kwargs,
            problem_type)
    end

    """
        ImmutableODEProblem{isinplace}(f,u0,tspan,p=NullParameters(),callback=CallbackSet())

    Define an ODE problem with the specified function.
    `isinplace` optionally sets whether the function is inplace or not.
    This is determined automatically, but not inferred.
    """
    function ImmutableODEProblem{iip}(f,
        u0,
        tspan,
        p = NullParameters();
        kwargs...) where {iip}
        _u0 = prepare_initial_state(u0)
        _tspan = promote_tspan(tspan)
        _f = ODEFunction{iip, DEFAULT_SPECIALIZATION}(f)
        ImmutableODEProblem(_f, _u0, _tspan, p; kwargs...)
    end

    @add_kwonly function ImmutableODEProblem{iip, recompile}(f, u0, tspan,
        p = NullParameters();
        kwargs...) where {iip, recompile}
        ImmutableODEProblem{iip}(ODEFunction{iip, recompile}(f), u0, tspan, p; kwargs...)
    end

    function ImmutableODEProblem{iip, FunctionWrapperSpecialize}(f, u0, tspan,
        p = NullParameters();
        kwargs...) where {iip}
        _u0 = prepare_initial_state(u0)
        _tspan = promote_tspan(tspan)
        if !(f isa FunctionWrappersWrappers.FunctionWrappersWrapper)
            if iip
                ff = ODEFunction{iip, FunctionWrapperSpecialize}(wrapfun_iip(f,
                    (_u0, _u0, p,
                        _tspan[1])))
            else
                ff = ODEFunction{iip, FunctionWrapperSpecialize}(wrapfun_oop(f,
                    (_u0, p,
                        _tspan[1])))
            end
        end
        ImmutableODEProblem{iip}(ff, _u0, _tspan, p; kwargs...)
    end
end

"""
    ImmutableODEProblem(f::ODEFunction,u0,tspan,p=NullParameters(),callback=CallbackSet())

Define an ODE problem from an [`ODEFunction`](@ref).
"""
function ImmutableODEProblem(f::AbstractODEFunction, u0, tspan, args...; kwargs...)
    ImmutableODEProblem{isinplace(f)}(f, u0, tspan, args...; kwargs...)
end

function ImmutableODEProblem(f, u0, tspan, p = NullParameters(); kwargs...)
    iip = isinplace(f, 4)
    _u0 = prepare_initial_state(u0)
    _tspan = promote_tspan(tspan)
    _f = ODEFunction{iip, DEFAULT_SPECIALIZATION}(f)
    ImmutableODEProblem(_f, _u0, _tspan, p; kwargs...)
end

staticarray_itize(x) = x
staticarray_itize(x::Vector) = SVector{length(x)}(x)
staticarray_itize(x::SizedVector) = SVector{length(x)}(x)
staticarray_itize(x::Matrix) = SMatrix{size(x)...}(x)
staticarray_itize(x::SizedMatrix) = SMatrix{size(x)...}(x)

function Base.convert(::Type{ImmutableODEProblem}, prob::T) where {T <: ODEProblem}
    ImmutableODEProblem(prob.f,
        staticarray_itize(prob.u0),
        prob.tspan,
        staticarray_itize(prob.p),
        prob.problem_type;
        prob.kwargs...)
end
