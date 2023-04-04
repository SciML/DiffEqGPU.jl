mutable struct GPURosenbrock23Integrator{IIP, S, T, ST, P, F, TS, CB, AlgType} <:
               DiffEqBase.AbstractODEIntegrator{AlgType, IIP, S, T}
    alg::AlgType
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
    callback::CB
    save_everystep::Bool
    saveat::ST
    cur_t::Int
    step_idx::Int
    event_last_time::Int
    vector_event_last_time::Int
    last_event_error::T
    k1::S                 #intepolants
    k2::S
    retcode::DiffEqBase.ReturnCode.T
end
const GPURB23I = GPURosenbrock23Integrator

@inline function (integrator::GPURosenbrock23Integrator)(t)
    Θ = (t - integrator.tprev) / integrator.dt
    _ode_interpolant(Θ, integrator.dt, integrator.uprev, integrator)
end

@inline function DiffEqBase.u_modified!(integrator::GPURosenbrock23Integrator, bool::Bool)
    integrator.u_modified = bool
end

@inline function gpurosenbrock23_init(alg::AlgType, f::F, IIP::Bool, u0::S, t0::T, dt::T,
                                      p::P, tstops::TS,
                                      callback::CB,
                                      save_everystep::Bool,
                                      saveat::ST) where {AlgType, F, P, T,
                                                         S <: AbstractArray{T},
                                                         TS, CB, ST}
    !IIP && @assert S <: SArray
    event_last_time = 1
    vector_event_last_time = 0
    last_event_error = zero(eltype(S))

    integ = GPURB23I{IIP, S, T, ST, P, F, TS, CB, AlgType}(alg, f, copy(u0), copy(u0),
                                                           copy(u0), t0, t0,
                                                           t0,
                                                           dt,
                                                           sign(dt), p, true, tstops, 1,
                                                           callback,
                                                           save_everystep, saveat, 1, 1,
                                                           event_last_time,
                                                           vector_event_last_time,
                                                           last_event_error,
                                                           copy(u0), copy(u0),
                                                           DiffEqBase.ReturnCode.Default)
end

mutable struct GPUARosenbrock23Integrator{IIP, S, T, ST, P, F, N, TOL, Q, TS, CB, AlgType
                                          } <:
               DiffEqBase.AbstractODEIntegrator{AlgType, IIP, S, T}
    alg::AlgType
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
    callback::CB
    save_everystep::Bool
    saveat::ST
    cur_t::Int
    step_idx::Int
    event_last_time::Int
    vector_event_last_time::Int
    last_event_error::T
    k1::S         # interpolants of the algorithm
    k2::S
    k3::S
    qold::Q
    abstol::TOL
    reltol::TOL
    internalnorm::N       # function that computes the error EEst based on state
    retcode::DiffEqBase.ReturnCode.T
end

const GPUARB23I = GPUARosenbrock23Integrator

@inline function (integrator::GPUARB23I)(t)
    Θ = (t - integrator.tprev) / integrator.dt
    _ode_interpolant(Θ, integrator.dt, integrator.uprev, integrator)
end

@inline function DiffEqBase.u_modified!(integrator::GPUARB23I, bool::Bool)
    integrator.u_modified = bool
end

@inline function gpuarosenbrock23_init(alg::AlgType, f::F, IIP::Bool, u0::S, t0::T, tf::T,
                                       dt::T, p::P,
                                       abstol::TOL, reltol::TOL,
                                       internalnorm::N, tstops::TS,
                                       callback::CB,
                                       saveat::ST) where {AlgType, F, P, S, T, N, TOL, TS,
                                                          CB, ST}
    !IIP && @assert S <: SArray

    qoldinit = eltype(S)(1e-4)
    event_last_time = 1
    vector_event_last_time = 0
    last_event_error = zero(eltype(S))

    integ = GPUARB23I{IIP, S, T, ST, P, F, N, TOL, typeof(qoldinit), TS, CB, AlgType}(alg,
                                                                                      f,
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
                                                                                      false,
                                                                                      saveat,
                                                                                      1, 1,
                                                                                      event_last_time,
                                                                                      vector_event_last_time,
                                                                                      last_event_error,
                                                                                      copy(u0),
                                                                                      copy(u0),
                                                                                      copy(u0),
                                                                                      qoldinit,
                                                                                      abstol,
                                                                                      reltol,
                                                                                      internalnorm,
                                                                                      DiffEqBase.ReturnCode.Default)
end
