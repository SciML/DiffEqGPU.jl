## Fixed TimeStep Integrator

function Adapt.adapt_structure(to, prob::ODEProblem{<:Any, <:Any, iip}) where {iip}
    ODEProblem{iip, true}(adapt(to, prob.f),
                          adapt(to, prob.u0),
                          adapt(to, prob.tspan),
                          adapt(to, prob.p);
                          adapt(to, prob.kwargs)...)
end

mutable struct GPUTsit5Integrator{IIP, S, T, P, F, TS, CB} <:
               DiffEqBase.AbstractODEIntegrator{GPUTsit5, IIP, S, T}
    f::F                  # eom
    uprev::S              # previous state
    u::S                  # current state
    tmp::S                # dummy, same as state
    tprev::T              # previous time
    t::T                  # current time
    t0::T                 # initial time, only for reinit
    dt::T                 # step size
    tdir::T
    p::P                  # parameter container
    u_modified::Bool
    tstops::TS
    tstops_idx::Int
    cb::CB
    save_everystep::Bool
    step_idx::Int
    k1::S                 #intepolants
    k2::S
    k3::S
    k4::S
    k5::S
    k6::S
    k7::S
    cs::SVector{6, T}     # ci factors cache: time coefficients
    as::SVector{21, T}    # aij factors cache: solution coefficients
    rs::SVector{22, T}    # rij factors cache: interpolation coefficients
end
const GPUT5I = GPUTsit5Integrator

(integrator::GPUTsit5Integrator)(t) = copy(integrator.u)
(integrator::GPUTsit5Integrator)(out, t) = (out .= integrator.u)

function DiffEqBase.u_modified!(integrator::GPUTsit5Integrator, bool::Bool)
    integrator.u_modified = bool
end

DiffEqBase.isinplace(::GPUT5I{IIP}) where {IIP} = IIP

## Adaptive TimeStep Integrator

mutable struct GPUATsit5Integrator{IIP, S, T, P, F, N, TOL, Q, TS, CB} <:
               DiffEqBase.AbstractODEIntegrator{GPUTsit5, IIP, S, T}
    f::F                  # eom
    uprev::S              # previous state
    u::S                  # current state
    tmp::S                # dummy, same as state
    tprev::T              # previous time
    t::T                  # current time
    t0::T                 # initial time, only for reinit
    tf::T
    dt::T                 # step size
    dtnew::T
    tdir::T
    p::P                  # parameter container
    u_modified::Bool
    tstops::TS
    tstops_idx::Int
    cb::CB
    k1::S         # interpolants of the algorithm
    k2::S
    k3::S
    k4::S
    k5::S
    k6::S
    k7::S
    cs::SVector{6, T}     # ci factors cache: time coefficients
    as::SVector{21, T}    # aij factors cache: solution coefficients
    btildes::SVector{7, T}
    rs::SVector{22, T}    # rij factors cache: interpolation coefficients
    qold::Q
    abstol::TOL
    reltol::TOL
    internalnorm::N       # function that computes the error EEst based on state
end

const GPUAT5I = GPUATsit5Integrator

## Vern7

mutable struct GPUVern7Integrator{IIP, S, T, P, F, TS, CB, TabType} <:
               DiffEqBase.AbstractODEIntegrator{GPUVern7, IIP, S, T}
    f::F                  # eom
    uprev::S              # previous state
    u::S                  # current state
    tmp::S                # dummy, same as state
    tprev::T              # previous time
    t::T                  # current time
    t0::T                 # initial time, only for reinit
    dt::T                 # step size
    tdir::T
    p::P                  # parameter container
    u_modified::Bool
    tstops::TS
    tstops_idx::Int
    cb::CB
    save_everystep::Bool
    step_idx::Int
    k1::S                 #intepolants
    k2::S
    k3::S
    k4::S
    k5::S
    k6::S
    k7::S
    k8::S
    k9::S
    k10::S
    tab::TabType
end
const GPUVern7I = GPUVern7Integrator

mutable struct GPUAVern7Integrator{IIP, S, T, P, F, N, TOL, Q, TS, CB, TabType} <:
               DiffEqBase.AbstractODEIntegrator{GPUVern7, IIP, S, T}
    f::F                  # eom
    uprev::S              # previous state
    u::S                  # current state
    tmp::S                # dummy, same as state
    tprev::T              # previous time
    t::T                  # current time
    t0::T                 # initial time, only for reinit
    tf::T
    dt::T                 # step size
    dtnew::T
    tdir::T
    p::P                  # parameter container
    u_modified::Bool
    tstops::TS
    tstops_idx::Int
    cb::CB
    k1::S         # interpolants of the algorithm
    k2::S
    k3::S
    k4::S
    k5::S
    k6::S
    k7::S
    k8::S
    k9::S
    k10::S
    tab::TabType
    qold::Q
    abstol::TOL
    reltol::TOL
    internalnorm::N       # function that computes the error EEst based on state
end

const GPUAVern7I = GPUAVern7Integrator

function (integrator::GPUAVern7I)(t)
    Θ = (t - integrator.tprev) / integrator.dt
    _ode_interpolant(Θ, integrator.dt, integrator.uprev, integ)
end

## Vern9

mutable struct GPUVern9Integrator{IIP, S, T, P, F, TS, CB, TabType} <:
               DiffEqBase.AbstractODEIntegrator{GPUVern9, IIP, S, T}
    f::F                  # eom
    uprev::S              # previous state
    u::S                  # current state
    tmp::S                # dummy, same as state
    tprev::T              # previous time
    t::T                  # current time
    t0::T                 # initial time, only for reinit
    dt::T                 # step size
    tdir::T
    p::P                  # parameter container
    u_modified::Bool
    tstops::TS
    tstops_idx::Int
    cb::CB
    save_everystep::Bool
    step_idx::Int
    k1::S                 #intepolants
    k2::S
    k3::S
    k4::S
    k5::S
    k6::S
    k7::S
    k8::S
    k9::S
    k10::S
    tab::TabType
end
const GPUVern9I = GPUVern9Integrator

mutable struct GPUAVern9Integrator{IIP, S, T, P, F, N, TOL, Q, TS, CB, TabType} <:
               DiffEqBase.AbstractODEIntegrator{GPUVern9, IIP, S, T}
    f::F                  # eom
    uprev::S              # previous state
    u::S                  # current state
    tmp::S                # dummy, same as state
    tprev::T              # previous time
    t::T                  # current time
    t0::T                 # initial time, only for reinit
    tf::T
    dt::T                 # step size
    dtnew::T
    tdir::T
    p::P                  # parameter container
    u_modified::Bool
    tstops::TS
    tstops_idx::Int
    cb::CB
    k1::S         # interpolants of the algorithm
    k2::S
    k3::S
    k4::S
    k5::S
    k6::S
    k7::S
    k8::S
    k9::S
    k10::S
    tab::TabType
    qold::Q
    abstol::TOL
    reltol::TOL
    internalnorm::N       # function that computes the error EEst based on state
end

const GPUAVern9I = GPUAVern9Integrator

function (integrator::GPUAVern9I)(t)
    Θ = (t - integrator.tprev) / integrator.dt
    _ode_interpolant(Θ, integrator.dt, integrator.uprev, integ)
end

#######################################################################################
# Initialization of Integrators
#######################################################################################
@inline function gputsit5_init(f::F, IIP::Bool, u0::S, t0::T, dt::T,
                               p::P, tstops::TS,
                               callback::CB,
                               save_everystep::Bool) where {F, P, T, S <: AbstractArray{T},
                                                            TS, CB}
    cs, as, rs = SimpleDiffEq._build_tsit5_caches(T)

    !IIP && @assert S <: SArray

    integ = GPUT5I{IIP, S, T, P, F, TS, CB}(f, copy(u0), copy(u0), copy(u0), t0, t0, t0, dt,
                                            sign(dt), p, true, tstops, 1, callback,
                                            save_everystep, 1,
                                            copy(u0), copy(u0), copy(u0), copy(u0),
                                            copy(u0),
                                            copy(u0), copy(u0), cs, as, rs)
end

@inline function gpuatsit5_init(f::F, IIP::Bool, u0::S, t0::T, tf::T, dt::T, p::P,
                                abstol::TOL, reltol::TOL,
                                internalnorm::N, tstops::TS,
                                callback::CB) where {F, P, S, T, N, TOL, TS, CB}
    cs, as, btildes, rs = SimpleDiffEq._build_atsit5_caches(T)

    !IIP && @assert S <: SArray

    qoldinit = eltype(S)(1e-4)

    integ = GPUAT5I{IIP, S, T, P, F, N, TOL, typeof(qoldinit), TS, CB}(f, copy(u0),
                                                                       copy(u0),
                                                                       copy(u0), t0, t0, t0,
                                                                       tf, dt,
                                                                       dt, sign(tf - t0), p,
                                                                       true, tstops, 1,
                                                                       callback,
                                                                       copy(u0), copy(u0),
                                                                       copy(u0),
                                                                       copy(u0), copy(u0),
                                                                       copy(u0),
                                                                       copy(u0), cs, as,
                                                                       btildes,
                                                                       rs, qoldinit, abstol,
                                                                       reltol,
                                                                       internalnorm)
end

@inline function gpuvern7_init(f::F, IIP::Bool, u0::S, t0::T, dt::T,
                               p::P, tstops::TS,
                               callback::CB,
                               save_everystep::Bool) where {F, P, T, S <: AbstractArray{T},
                                                            TS, CB}
    tab = Vern7Tableau(T, T)

    !IIP && @assert S <: SArray

    integ = GPUVern7I{IIP, S, T, P, F, TS, CB, typeof(tab)}(f, copy(u0), copy(u0), copy(u0),
                                                            t0, t0, t0, dt,
                                                            sign(dt), p, true, tstops, 1,
                                                            callback,
                                                            save_everystep, 1,
                                                            copy(u0), copy(u0), copy(u0),
                                                            copy(u0),
                                                            copy(u0),
                                                            copy(u0), copy(u0), copy(u0),
                                                            copy(u0), copy(u0), tab)
end

@inline function gpuavern7_init(f::F, IIP::Bool, u0::S, t0::T, tf::T, dt::T, p::P,
                                abstol::TOL, reltol::TOL,
                                internalnorm::N, tstops::TS,
                                callback::CB) where {F, P, S, T, N, TOL, TS, CB}
    !IIP && @assert S <: SArray

    tab = Vern7Tableau(T, T)

    qoldinit = eltype(S)(1e-4)

    integ = GPUAVern7I{IIP, S, T, P, F, N, TOL, typeof(qoldinit), TS, CB, typeof(tab)}(f,
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       t0,
                                                                                       t0,
                                                                                       t0,
                                                                                       tf,
                                                                                       dt,
                                                                                       dt,
                                                                                       sign(tf -
                                                                                            t0),
                                                                                       p,
                                                                                       true,
                                                                                       tstops,
                                                                                       1,
                                                                                       callback,
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       tab,
                                                                                       qoldinit,
                                                                                       abstol,
                                                                                       reltol,
                                                                                       internalnorm)
end

@inline function gpuvern9_init(f::F, IIP::Bool, u0::S, t0::T, dt::T,
                               p::P, tstops::TS,
                               callback::CB,
                               save_everystep::Bool) where {F, P, T, S <: AbstractArray{T},
                                                            TS, CB}
    tab = Vern9Tableau(T, T)

    !IIP && @assert S <: SArray

    integ = GPUVern9I{IIP, S, T, P, F, TS, CB, typeof(tab)}(f, copy(u0), copy(u0), copy(u0),
                                                            t0, t0, t0, dt,
                                                            sign(dt), p, true, tstops, 1,
                                                            callback,
                                                            save_everystep, 1,
                                                            copy(u0), copy(u0), copy(u0),
                                                            copy(u0), copy(u0), copy(u0),
                                                            copy(u0), copy(u0),
                                                            copy(u0), copy(u0), tab)
end

@inline function gpuavern9_init(f::F, IIP::Bool, u0::S, t0::T, tf::T, dt::T, p::P,
                                abstol::TOL, reltol::TOL,
                                internalnorm::N, tstops::TS,
                                callback::CB) where {F, P, S, T, N, TOL, TS, CB}
    !IIP && @assert S <: SArray

    tab = Vern9Tableau(T, T)

    qoldinit = eltype(S)(1e-4)

    integ = GPUAVern9I{IIP, S, T, P, F, N, TOL, typeof(qoldinit), TS, CB, typeof(tab)}(f,
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       t0,
                                                                                       t0,
                                                                                       t0,
                                                                                       tf,
                                                                                       dt,
                                                                                       dt,
                                                                                       sign(tf -
                                                                                            t0),
                                                                                       p,
                                                                                       true,
                                                                                       tstops,
                                                                                       1,
                                                                                       callback,
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       copy(u0),
                                                                                       tab,
                                                                                       qoldinit,
                                                                                       abstol,
                                                                                       reltol,
                                                                                       internalnorm)
end
