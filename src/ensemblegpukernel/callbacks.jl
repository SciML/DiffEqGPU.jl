struct GPUDiscreteCallback{F1, F2, F3, F4, F5} <: SciMLBase.AbstractDiscreteCallback
    condition::F1
    affect!::F2
    initialize::F3
    finalize::F4
    save_positions::F5
    function GPUDiscreteCallback(condition::F1, affect!::F2,
            initialize::F3, finalize::F4,
            save_positions::F5) where {F1, F2, F3, F4, F5}
        if save_positions != (false, false)
            error("Callback `save_positions` are incompatible with kernel-based GPU ODE solvers due requiring static sizing. Please ensure `save_positions = (false,false)` is set in all callback definitions used with such solvers.")
        end
        new{F1, F2, F3, F4, F5}(condition,
            affect!, initialize, finalize, save_positions)
    end
end
function GPUDiscreteCallback(condition, affect!;
        initialize = SciMLBase.INITIALIZE_DEFAULT,
        finalize = SciMLBase.FINALIZE_DEFAULT,
        save_positions = (false, false))
    GPUDiscreteCallback(condition, affect!, initialize, finalize, save_positions)
end

function Base.convert(::Type{GPUDiscreteCallback}, x::T) where {T <: DiscreteCallback}
    GPUDiscreteCallback(x.condition, x.affect!, x.initialize, x.finalize,
        Tuple(x.save_positions))
end

struct GPUContinuousCallback{F1, F2, F3, F4, F5, F6, T, T2, T3, I, R} <:
       SciMLBase.AbstractContinuousCallback
    condition::F1
    affect!::F2
    affect_neg!::F3
    initialize::F4
    finalize::F5
    idxs::I
    rootfind::SciMLBase.RootfindOpt
    interp_points::Int
    save_positions::F6
    dtrelax::R
    abstol::T
    reltol::T2
    repeat_nudge::T3
    function GPUContinuousCallback(condition::F1, affect!::F2, affect_neg!::F3,
            initialize::F4, finalize::F5, idxs::I, rootfind,
            interp_points, save_positions::F6, dtrelax::R, abstol::T,
            reltol::T2,
            repeat_nudge::T3) where {F1, F2, F3, F4, F5, F6, T, T2,
            T3, I, R,
        }
        if save_positions != (false, false)
            error("Callback `save_positions` are incompatible with kernel-based GPU ODE solvers due requiring static sizing. Please ensure `save_positions = (false,false)` is set in all callback definitions used with such solvers.")
        end
        new{F1, F2, F3, F4, F5, F6, T, T2, T3, I, R}(condition,
            affect!, affect_neg!,
            initialize, finalize, idxs, rootfind,
            interp_points,
            save_positions,
            dtrelax, abstol, reltol, repeat_nudge)
    end
end

function GPUContinuousCallback(condition, affect!, affect_neg!;
        initialize = SciMLBase.INITIALIZE_DEFAULT,
        finalize = SciMLBase.FINALIZE_DEFAULT,
        idxs = nothing,
        rootfind = LeftRootFind,
        save_positions = (false, false),
        interp_points = 10,
        dtrelax = 1,
        abstol = 10eps(Float32), reltol = 0,
        repeat_nudge = 1 // 100)
    GPUContinuousCallback(condition, affect!, affect_neg!, initialize, finalize,
        idxs,
        rootfind, interp_points,
        save_positions,
        dtrelax, abstol, reltol, repeat_nudge)
end

function GPUContinuousCallback(condition, affect!;
        initialize = SciMLBase.INITIALIZE_DEFAULT,
        finalize = SciMLBase.FINALIZE_DEFAULT,
        idxs = nothing,
        rootfind = LeftRootFind,
        save_positions = (false, false),
        affect_neg! = affect!,
        interp_points = 10,
        dtrelax = 1,
        abstol = 10eps(Float32), reltol = 0, repeat_nudge = 1 // 100)
    GPUContinuousCallback(condition, affect!, affect_neg!, initialize, finalize, idxs,
        rootfind, interp_points,
        save_positions,
        dtrelax, abstol, reltol, repeat_nudge)
end

function Base.convert(::Type{GPUContinuousCallback}, x::T) where {T <: ContinuousCallback}
    GPUContinuousCallback(x.condition, x.affect!, x.affect_neg!, x.initialize, x.finalize,
        x.idxs, x.rootfind, x.interp_points,
        Tuple(x.save_positions), x.dtrelax, 100 * eps(Float32), x.reltol,
        x.repeat_nudge)
end

function generate_callback(callback::DiscreteCallback, I,
        ensemblealg)
    if ensemblealg isa EnsembleGPUArray
        backend = ensemblealg.backend
        cur = adapt(backend, [false for i in 1:I])
    elseif ensemblealg isa EnsembleGPUKernel
        return callback
    else
        cur = [false for i in 1:I]
    end
    _condition = callback.condition
    _affect! = callback.affect!

    condition = function (u, t, integrator)
        version = get_backend(u)
        wgs = workgroupsize(version, size(u, 2))
        discrete_condition_kernel(version)(_condition, cur, u, t, integrator.p;
            ndrange = size(u, 2),
            workgroupsize = wgs)
        any(cur)
    end

    affect! = function (integrator)
        version = get_backend(integrator.u)
        wgs = workgroupsize(version, size(integrator.u, 2))
        discrete_affect!_kernel(version)(_affect!, cur, integrator.u, integrator.t,
            integrator.p;
            ndrange = size(integrator.u, 2),
            workgroupsize = wgs)
    end
    return DiscreteCallback(condition, affect!, save_positions = callback.save_positions)
end
