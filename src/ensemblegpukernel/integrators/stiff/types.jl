@inline function (
        integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP,
            S,
            T,
        }
    )(t) where {
        AlgType <:
        GPUODEAlgorithm,
        IIP,
        S,
        T,
    }
    Θ = (t - integrator.tprev) / integrator.dt
    return _ode_interpolant(Θ, integrator.dt, integrator.uprev, integrator)
end

@inline function DiffEqBase.u_modified!(
        integrator::DiffEqBase.AbstractODEIntegrator{
            AlgType,
            IIP, S,
            T,
        },
        bool::Bool
    ) where {
        AlgType <: GPUODEAlgorithm, IIP,
        S, T,
    }
    return integrator.u_modified = bool
end

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
    d::T
    retcode::DiffEqBase.ReturnCode.T
end
const GPURB23I = GPURosenbrock23Integrator

@inline function init(
        alg::GPURosenbrock23, f::F, IIP::Bool, u0::S, t0::T, dt::T,
        p::P, tstops::TS,
        callback::CB,
        save_everystep::Bool,
        saveat::ST
    ) where {
        F, P, T,
        S,
        TS, CB, ST,
    }
    !IIP && @assert S <: SArray
    event_last_time = 1
    vector_event_last_time = 0
    last_event_error = zero(T)
    d = T(2)
    d = 1 / (d + sqrt(d))

    return integ = GPURB23I{IIP, S, T, ST, P, F, TS, CB, typeof(alg)}(
        alg, f, copy(u0), copy(u0),
        copy(u0), t0, t0,
        t0,
        dt,
        sign(dt), p, true, tstops, 1,
        callback,
        save_everystep, saveat, 1, 1,
        event_last_time,
        vector_event_last_time,
        last_event_error,
        copy(u0), copy(u0), d,
        DiffEqBase.ReturnCode.Default
    )
end

mutable struct GPUARosenbrock23Integrator{
        IIP,
        S,
        T,
        ST,
        P,
        F,
        N,
        TOL,
        Q,
        TS,
        CB,
        AlgType,
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
    d::T
    qold::Q
    abstol::TOL
    reltol::TOL
    internalnorm::N       # function that computes the error EEst based on state
    retcode::DiffEqBase.ReturnCode.T
end

const GPUARB23I = GPUARosenbrock23Integrator

@inline function init(
        alg::GPURosenbrock23, f::F, IIP::Bool, u0::S, t0::T, tf::T,
        dt::T, p::P,
        abstol::TOL, reltol::TOL,
        internalnorm::N, tstops::TS,
        callback::CB,
        saveat::ST
    ) where {
        F, P, S, T, N, TOL, TS,
        CB, ST,
    }
    !IIP && @assert S <: SArray

    qoldinit = T(1.0e-4)
    event_last_time = 1
    vector_event_last_time = 0
    last_event_error = zero(T)

    d = T(2)
    d = 1 / (d + sqrt(d))

    return integ = GPUARB23I{IIP, S, T, ST, P, F, N, TOL, typeof(qoldinit), TS, CB, typeof(alg)}(
        alg,
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
        sign(
            tf -
                t0
        ),
        p,
        true,
        tstops,
        1,
        callback,
        false,
        saveat,
        1,
        1,
        event_last_time,
        vector_event_last_time,
        last_event_error,
        copy(u0),
        copy(u0),
        copy(u0),
        d,
        qoldinit,
        abstol,
        reltol,
        internalnorm,
        DiffEqBase.ReturnCode.Default
    )
end

##########################
# Rodas 4
##########################
# Fixed Step
mutable struct GPURodas4Integrator{IIP, S, T, ST, P, F, TS, CB, TabType, AlgType} <:
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
    k2::S                 #intepolants
    tab::TabType
    retcode::DiffEqBase.ReturnCode.T
end
const GPURodas4I = GPURodas4Integrator

@inline function init(
        alg::GPURodas4, f::F, IIP::Bool, u0::S, t0::T, dt::T,
        p::P, tstops::TS,
        callback::CB,
        save_everystep::Bool,
        saveat::ST
    ) where {
        F, P, T,
        S,
        TS, CB, ST,
    }
    !IIP && @assert S <: SArray
    event_last_time = 1
    vector_event_last_time = 0
    last_event_error = zero(T)

    tab = Rodas4Tableau(T, T)

    return integ = GPURodas4I{IIP, S, T, ST, P, F, TS, CB, typeof(tab), typeof(alg)}(
        alg, f,
        copy(u0),
        copy(u0),
        copy(u0), t0,
        t0,
        t0,
        dt,
        sign(dt), p,
        true,
        tstops, 1,
        callback,
        save_everystep,
        saveat, 1, 1,
        event_last_time,
        vector_event_last_time,
        last_event_error,
        copy(u0),
        copy(u0), tab,
        DiffEqBase.ReturnCode.Default
    )
end

# Adaptive Step
mutable struct GPUARodas4Integrator{
        IIP,
        S,
        T,
        ST,
        P,
        F,
        N,
        TOL,
        Q,
        TS,
        CB,
        TabType,
        AlgType,
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
    k1::S               #intepolants
    k2::S               #intepolants
    tab::TabType
    qold::Q
    abstol::TOL
    reltol::TOL
    internalnorm::N       # function that computes the error EEst based on state
    retcode::DiffEqBase.ReturnCode.T
end

const GPUARodas4I = GPUARodas4Integrator

@inline function init(
        alg::GPURodas4, f::F, IIP::Bool, u0::S, t0::T, tf::T,
        dt::T, p::P,
        abstol::TOL, reltol::TOL,
        internalnorm::N, tstops::TS,
        callback::CB,
        saveat::ST
    ) where {
        F, P, S, T, N, TOL, TS,
        CB, ST,
    }
    !IIP && @assert S <: SArray
    qoldinit = T(1.0e-4)
    event_last_time = 1
    vector_event_last_time = 0
    last_event_error = zero(T)

    tab = Rodas4Tableau(T, T)

    return integ = GPUARodas4I{
        IIP, S, T, ST, P, F, N, TOL, typeof(qoldinit), TS, CB, typeof(tab),
        typeof(alg),
    }(
        alg,
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
        sign(
            tf -
                t0
        ),
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
        tab,
        qoldinit,
        abstol,
        reltol,
        internalnorm,
        DiffEqBase.ReturnCode.Default
    )
end

##########################
# Rodas 5P
##########################
# Fixed Step
mutable struct GPURodas5PIntegrator{IIP, S, T, ST, P, F, TS, CB, TabType, AlgType} <:
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
    k2::S                 #intepolants
    k3::S                 #intepolants
    tab::TabType
    retcode::DiffEqBase.ReturnCode.T
end
const GPURodas5PI = GPURodas5PIntegrator

@inline function init(
        alg::GPURodas5P, f::F, IIP::Bool, u0::S, t0::T, dt::T,
        p::P, tstops::TS,
        callback::CB,
        save_everystep::Bool,
        saveat::ST
    ) where {
        F, P, T,
        S,
        TS, CB, ST,
    }
    !IIP && @assert S <: SArray
    event_last_time = 1
    vector_event_last_time = 0
    last_event_error = zero(T)

    tab = Rodas5PTableau(T, T)

    return integ = GPURodas5PI{IIP, S, T, ST, P, F, TS, CB, typeof(tab), typeof(alg)}(
        alg, f,
        copy(u0),
        copy(u0),
        copy(u0), t0,
        t0,
        t0,
        dt,
        sign(dt), p,
        true,
        tstops, 1,
        callback,
        save_everystep,
        saveat, 1, 1,
        event_last_time,
        vector_event_last_time,
        last_event_error,
        copy(u0),
        copy(u0),
        copy(u0),
        tab,
        DiffEqBase.ReturnCode.Default
    )
end

# Adaptive Step
mutable struct GPUARodas5PIntegrator{
        IIP, S, T, ST, P, F, N, TOL, Q, TS, CB, TabType,
        AlgType,
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
    k1::S               #intepolants
    k2::S               #intepolants
    k3::S                 #intepolants
    tab::TabType
    qold::Q
    abstol::TOL
    reltol::TOL
    internalnorm::N       # function that computes the error EEst based on state
    retcode::DiffEqBase.ReturnCode.T
end

const GPUARodas5PI = GPUARodas5PIntegrator

@inline function init(
        alg::GPURodas5P, f::F, IIP::Bool, u0::S, t0::T, tf::T,
        dt::T, p::P,
        abstol::TOL, reltol::TOL,
        internalnorm::N, tstops::TS,
        callback::CB,
        saveat::ST
    ) where {
        F, P, S, T, N, TOL, TS,
        CB, ST,
    }
    !IIP && @assert S <: SArray
    qoldinit = T(1.0e-4)
    event_last_time = 1
    vector_event_last_time = 0
    last_event_error = zero(T)

    tab = Rodas5PTableau(T, T)

    return integ = GPUARodas5PI{
        IIP, S, T, ST, P, F, N, TOL, typeof(qoldinit), TS, CB, typeof(tab),
        typeof(alg),
    }(
        alg,
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
        sign(
            tf -
                t0
        ),
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
        tab,
        qoldinit,
        abstol,
        reltol,
        internalnorm,
        DiffEqBase.ReturnCode.Default
    )
end

##########################
# Kvaerno3
##########################
# Fixed Step
mutable struct GPUKvaerno3Integrator{IIP, S, T, ST, P, F, TS, CB, TabType, AlgType} <:
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
    k2::S                 #intepolants
    tab::TabType
    retcode::DiffEqBase.ReturnCode.T
end
const GPUKvaerno3I = GPUKvaerno3Integrator

@inline function init(
        alg::GPUKvaerno3, f::F, IIP::Bool, u0::S, t0::T, dt::T,
        p::P, tstops::TS,
        callback::CB,
        save_everystep::Bool,
        saveat::ST
    ) where {
        F, P, T,
        S,
        TS, CB, ST,
    }
    !IIP && @assert S <: SArray
    event_last_time = 1
    vector_event_last_time = 0
    last_event_error = zero(T)

    tab = Kvaerno3Tableau(T, T)

    return integ = GPUKvaerno3I{IIP, S, T, ST, P, F, TS, CB, typeof(tab), typeof(alg)}(
        alg, f,
        copy(u0),
        copy(u0),
        copy(u0),
        t0,
        t0,
        t0,
        dt,
        sign(dt), p,
        true,
        tstops, 1,
        callback,
        save_everystep,
        saveat, 1,
        1,
        event_last_time,
        vector_event_last_time,
        last_event_error,
        copy(u0),
        copy(u0),
        tab,
        DiffEqBase.ReturnCode.Default
    )
end

# Adaptive Step
mutable struct GPUAKvaerno3Integrator{
        IIP, S, T, ST, P, F, N, TOL, Q, TS, CB, TabType,
        AlgType,
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
    k1::S               #intepolants
    k2::S               #intepolants
    tab::TabType
    qold::Q
    abstol::TOL
    reltol::TOL
    internalnorm::N       # function that computes the error EEst based on state
    retcode::DiffEqBase.ReturnCode.T
end

const GPUAKvaerno3I = GPUAKvaerno3Integrator

@inline function init(
        alg::GPUKvaerno3, f::F, IIP::Bool, u0::S, t0::T, tf::T,
        dt::T, p::P,
        abstol::TOL, reltol::TOL,
        internalnorm::N, tstops::TS,
        callback::CB,
        saveat::ST
    ) where {
        F, P, S, T, N, TOL, TS,
        CB, ST,
    }
    !IIP && @assert S <: SArray
    qoldinit = T(1.0e-4)
    event_last_time = 1
    vector_event_last_time = 0
    last_event_error = zero(T)

    tab = Kvaerno3Tableau(T, T)

    return integ = GPUAKvaerno3I{
        IIP, S, T, ST, P, F, N, TOL, typeof(qoldinit), TS, CB,
        typeof(tab),
        typeof(alg),
    }(
        alg,
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
        sign(
            tf -
                t0
        ),
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
        tab,
        qoldinit,
        abstol,
        reltol,
        internalnorm,
        DiffEqBase.ReturnCode.Default
    )
end

##########################
# Kvaerno5
##########################
# Fixed Step
mutable struct GPUKvaerno5Integrator{IIP, S, T, ST, P, F, TS, CB, TabType, AlgType} <:
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
    k2::S                 #intepolants
    tab::TabType
    retcode::DiffEqBase.ReturnCode.T
end
const GPUKvaerno5I = GPUKvaerno5Integrator

@inline function init(
        alg::GPUKvaerno5, f::F, IIP::Bool, u0::S, t0::T, dt::T,
        p::P, tstops::TS,
        callback::CB,
        save_everystep::Bool,
        saveat::ST
    ) where {
        F, P, T,
        S,
        TS, CB, ST,
    }
    !IIP && @assert S <: SArray
    event_last_time = 1
    vector_event_last_time = 0
    last_event_error = zero(T)

    tab = Kvaerno5Tableau(T, T)

    return integ = GPUKvaerno5I{IIP, S, T, ST, P, F, TS, CB, typeof(tab), typeof(alg)}(
        alg, f,
        copy(u0),
        copy(u0),
        copy(u0),
        t0,
        t0,
        t0,
        dt,
        sign(dt), p,
        true,
        tstops, 1,
        callback,
        save_everystep,
        saveat, 1,
        1,
        event_last_time,
        vector_event_last_time,
        last_event_error,
        copy(u0),
        copy(u0),
        tab,
        DiffEqBase.ReturnCode.Default
    )
end

# Adaptive Step
mutable struct GPUAKvaerno5Integrator{
        IIP, S, T, ST, P, F, N, TOL, Q, TS, CB, TabType,
        AlgType,
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
    k1::S               #intepolants
    k2::S               #intepolants
    tab::TabType
    qold::Q
    abstol::TOL
    reltol::TOL
    internalnorm::N       # function that computes the error EEst based on state
    retcode::DiffEqBase.ReturnCode.T
end

const GPUAKvaerno5I = GPUAKvaerno5Integrator

@inline function init(
        alg::GPUKvaerno5, f::F, IIP::Bool, u0::S, t0::T, tf::T,
        dt::T, p::P,
        abstol::TOL, reltol::TOL,
        internalnorm::N, tstops::TS,
        callback::CB,
        saveat::ST
    ) where {
        F, P, S, T, N, TOL, TS,
        CB, ST,
    }
    !IIP && @assert S <: SArray
    qoldinit = T(1.0e-4)
    event_last_time = 1
    vector_event_last_time = 0
    last_event_error = zero(T)

    tab = Kvaerno5Tableau(T, T)

    return integ = GPUAKvaerno5I{
        IIP, S, T, ST, P, F, N, TOL, typeof(qoldinit), TS, CB,
        typeof(tab),
        typeof(alg),
    }(
        alg,
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
        sign(
            tf -
                t0
        ),
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
        tab,
        qoldinit,
        abstol,
        reltol,
        internalnorm,
        DiffEqBase.ReturnCode.Default
    )
end
