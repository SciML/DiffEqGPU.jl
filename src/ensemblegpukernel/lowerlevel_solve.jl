"""
    vectorized_solve(
        probs, prob::Union{ODEProblem, SDEProblem}, alg; dt,
        saveat = nothing, save_everystep = true, debug = false,
        callback = CallbackSet(nothing), tstops = nothing
    )

Run a fixed-step `EnsembleGPUKernel` solve on a batch of compatible problems. This is a
developer-facing entry point for packages that need the batched time and state arrays
instead of a collection of `EnsembleSolution`s.

# Arguments

  - `probs`: a batch of problems adapted to the backend returned by `get_backend(probs)`.
    For ODE problems, each element must be compatible with the static, GPU-compilable
    kernel representation.
  - `prob`: a representative `ODEProblem` or `SDEProblem` whose time span and state type
    determine the output layout. For an ensemble batch this is normally the original
    problem or `probs[1]`.
  - `alg`: a supported `GPUODEAlgorithm` or `GPUSDEAlgorithm`, such as `GPUTsit5()` or
    `GPUEM()`.

# Keyword Arguments

  - `dt`: required fixed time step.
  - `saveat`: optional scalar, vector, or range of output times. `nothing` uses the regular
    time grid implied by `dt`.
  - `save_everystep`: whether to retain every fixed-step state when `saveat === nothing`.
  - `debug`: reserved debugging option; it is accepted for compatibility with the solver
    interface.
  - `callback`: a GPU-compatible callback set. The default is `CallbackSet(nothing)`.
  - `tstops`: optional additional stopping times.

# Returns

A pair `(ts, us)`. `ts` contains the saved times and `us` contains the corresponding
states, with one batch trajectory per column. The arrays remain on the selected backend.

# Throws

An `ArgumentError` or `MethodError` can be raised when the batch, problem, algorithm, or
callback is not compatible with GPU kernel execution. SDE algorithms also throw when the
noise structure is unsupported.

# Examples

```julia
ts, us = DiffEqGPU.vectorized_solve(
    gpu_probs, prob, GPUTsit5(); dt = 0.1f0, save_everystep = false
)
```
"""
function vectorized_solve end


# Pack AbstractFloat kernel scalars as 0-d device arrays so Enzyme sees Duplicated,
# not Active, GPU args. Duals and non-floats pass through unchanged.
function _pack_kernel_scalar(backend, x::AbstractFloat)
    a = allocate(backend, typeof(x), ())
    _init_time_matrix!(a, x)
    return a
end
_pack_kernel_scalar(backend, x) = x

# Time matrix init must use tspan[1] as the unused-slot sentinel (see findlast in
# batch_solve). Under Enzyme, Active fill! scalars are rejected on GPU — mark inactive.
@noinline function _init_time_matrix!(ts, t0)
    fill!(ts, t0)
    return ts
end


function vectorized_solve(
        probs, prob::ODEProblem, alg;
        dt, saveat = nothing,
        save_everystep = true,
        debug = false, callback = CallbackSet(nothing), tstops = nothing,
        kwargs...
    )
    backend = get_backend(probs)
    backend = maybe_prefer_blocks(backend)

    # Avoid StepRangeLen: Enzyme reverse and some Dual GPU backends mishandle it.
    prob = convert(ImmutableODEProblem, prob)
    dt = convert(eltype(prob.tspan), dt)
    saveat_converted = nothing

    if saveat === nothing
        if save_everystep
            len = ceil(Int, abs(prob.tspan[2] - prob.tspan[1]) / abs(dt)) + 1
            if tstops !== nothing
                len += length(tstops)
            end
        else
            len = 2
        end
        ts = allocate(backend, typeof(dt), (len, length(probs)))
        _init_time_matrix!(ts, prob.tspan[1])
        us = allocate(backend, typeof(prob.u0), (len, length(probs)))
    else
        Tt = eltype(prob.tspan)
        saveat_converted = if saveat isa AbstractRange
            Tt.(collect(range(Tt(first(saveat)), Tt(last(saveat)), length = length(saveat))))
        elseif saveat isa AbstractVector
            Tt.(collect(saveat))
        else
            t0, tf = Tt.(prob.tspan)
            if Tt(saveat) == Tt(0.0)
                Tt.([t0, tf])
            else
                num_points = Int(ceil(abs(tf - t0) / abs(Tt(saveat)))) + 1
                Tt.(collect(range(t0, tf, length = num_points)))
            end
        end
        saveat_converted = adapt(backend, saveat_converted)
        ts = allocate(backend, typeof(dt), (length(saveat_converted), length(probs)))
        _init_time_matrix!(ts, prob.tspan[1])
        us = allocate(backend, typeof(prob.u0), (length(saveat_converted), length(probs)))
    end

    tstops = adapt(backend, tstops)

    if backend isa CPU
        @warn "Running the kernel on CPU"
    end

    ode_solve_kernel(backend)(
        probs, alg, us, ts, _pack_kernel_scalar(backend, dt), callback, tstops,
        saveat_converted, Val(save_everystep);
        ndrange = length(probs)
    )

    return ts, us
end

# SDEProblems over GPU cannot support u0 as a Number type, because GPU kernels compiled only through u0 being StaticArrays
function vectorized_solve(
        probs, prob::SDEProblem, alg;
        dt, saveat = nothing,
        save_everystep = true,
        debug = false,
        kwargs...
    )
    backend = get_backend(probs)
    backend = maybe_prefer_blocks(backend)

    dt = convert(eltype(prob.tspan), dt)
    saveat_converted = nothing
    if saveat === nothing
        if save_everystep
            len = length(prob.tspan[1]:dt:prob.tspan[2])
        else
            len = 2
        end
        ts = allocate(backend, typeof(dt), (len, length(probs)))
        _init_time_matrix!(ts, prob.tspan[1])
        us = allocate(backend, typeof(prob.u0), (len, length(probs)))
    else
        # Get the time type from the problem
        Tt = eltype(prob.tspan)

        # FIX for Issue #379: Convert saveat to proper type
        saveat_converted = if saveat isa AbstractRange
            Tt.(collect(range(Tt(first(saveat)), Tt(last(saveat)), length = length(saveat))))
        elseif saveat isa AbstractVector
            Tt.(collect(saveat))
        else
            # saveat is a Number (step size)
            t0, tf = Tt.(prob.tspan)
            if Tt(saveat) == Tt(0.0)
                Tt.([t0, tf])
            else
                num_points = Int(ceil(abs(tf - t0) / abs(Tt(saveat)))) + 1
                Tt.(collect(range(t0, tf, length = num_points)))
            end
        end

        ts = allocate(backend, typeof(dt), (length(saveat_converted), length(probs)))
        _init_time_matrix!(ts, prob.tspan[1])
        us = allocate(backend, typeof(prob.u0), (length(saveat_converted), length(probs)))
    end
    if saveat_converted !== nothing
        saveat_converted = adapt(backend, saveat_converted)
    end
    kernel = if alg isa GPUEM
        em_kernel(backend)
    elseif alg isa Union{GPUSIEA}
        SciMLBase.is_diagonal_noise(prob) ? nothing :
            error("The algorithm is not compatible with the chosen noise type. Please see the documentation on the solver methods")
        siea_kernel(backend)
    else
        error("The algorithm is not compatible with the chosen problem type. Please see the documentation on the solver methods")
    end

    if backend isa CPU
        @warn "Running the kernel on CPU"
    end

    kernel(
        probs, us, ts, dt, saveat_converted, Val(save_everystep);
        ndrange = length(probs)
    )
    return ts, us
end

"""
    vectorized_asolve(
        probs, prob::ODEProblem, alg; dt = 0.1f0,
        saveat = nothing, save_everystep = false, abstol = 1.0f-6,
        reltol = 1.0f-3, debug = false, callback = CallbackSet(nothing),
        tstops = nothing
    )

Run an adaptive `EnsembleGPUKernel` ODE solve on a batch of compatible problems. This is a
developer-facing entry point for packages that need the batched time and state arrays
instead of a collection of `EnsembleSolution`s.

# Arguments

  - `probs`: a batch of ODE problems adapted to the backend returned by
    `get_backend(probs)`. Each element must be compatible with the static,
    GPU-compilable kernel representation.
  - `prob`: a representative `ODEProblem` whose time span and state type determine the
    output layout. For an ensemble batch this is normally the original problem or
    `probs[1]`.
  - `alg`: a supported `GPUODEAlgorithm`, such as `GPUTsit5()` or `GPURodas4()`.

# Keyword Arguments

  - `dt`: initial time step; defaults to `0.1f0`.
  - `saveat`: optional scalar, vector, or range of output times. `nothing` uses adaptive
    internal steps and the value of `save_everystep`.
  - `save_everystep`: whether to retain the adaptive internal steps. Defaults to `false`.
  - `abstol`: absolute error tolerance. Defaults to `1.0f-6`.
  - `reltol`: relative error tolerance. Defaults to `1.0f-3`.
  - `debug`: accepted for compatibility with the vectorized solver interface.
  - `callback`: a GPU-compatible callback set. The default is `CallbackSet(nothing)`.
  - `tstops`: optional additional stopping times.

# Returns

A pair `(ts, us)` containing the saved times and states, with one batch trajectory per
column. The arrays remain on the selected backend.

# Throws

An `ArgumentError` or `MethodError` can be raised when the batch, problem, algorithm, or
callback is not compatible with adaptive GPU kernel execution.

# Examples

```julia
ts, us = DiffEqGPU.vectorized_asolve(
    gpu_probs, prob, GPUTsit5(); dt = 0.1f0, abstol = 1.0f-6, reltol = 1.0f-3
)
```
"""
function vectorized_asolve end

function vectorized_asolve(
        probs, prob::ODEProblem, alg;
        dt = 0.1f0, saveat = nothing,
        save_everystep = false,
        abstol = 1.0f-6, reltol = 1.0f-3,
        debug = false, callback = CallbackSet(nothing), tstops = nothing,
        kwargs...
    )

    backend = get_backend(probs)
    backend = maybe_prefer_blocks(backend)

    # Get the time type from the problem
    Tt = eltype(prob.tspan)

    # FIX for Issue #379: Convert saveat to eliminate
    # StepRangeLen's internal Float64 fields which crash Metal

    if saveat !== nothing
        if saveat isa Number
            # Handle edge case: saveat = 0.0 means only save endpoints
            if Tt(saveat) == Tt(0.0)
                saveat_converted = Tt.([prob.tspan[1], prob.tspan[2]])
            else
                # Create proper range with correct type
                t0, tf = Tt.(prob.tspan)

                # Handle both forward and reverse time integration
                num_points = Int(ceil(abs(tf - t0) / abs(Tt(saveat)))) + 1

                # Safety check: prevent massive arrays
                max_saveat_length = 100_000
                if num_points > max_saveat_length
                    error(
                        "saveat would create too many save points ($num_points). " *
                            "Consider using a larger saveat value."
                    )
                end

                # Create range and convert to pure Vector{Tt}
                saveat_range = range(t0, tf, length = num_points)
                saveat_converted = Tt.(collect(saveat_range))
            end
        elseif saveat isa AbstractRange || saveat isa AbstractArray
            # Range or array - convert all elements to Tt
            # This eliminates StepRangeLen's Float64 internals
            saveat_converted = Tt.(collect(saveat))
        else
            # Already in correct form
            saveat_converted = saveat
        end
    else
        saveat_converted = nothing
    end

    prob = convert(ImmutableODEProblem, prob)
    dt = convert(eltype(prob.tspan), dt)

    if saveat_converted === nothing
        if save_everystep
            len = ceil(Int, (prob.tspan[2] - prob.tspan[1]) / dt) + 1
        else
            len = 2
        end
        ts = allocate(backend, typeof(dt), (len, length(probs)))
        _init_time_matrix!(ts, prob.tspan[1])
        us = allocate(backend, typeof(prob.u0), (len, length(probs)))
    else
        ts = allocate(backend, typeof(dt), (length(saveat_converted), length(probs)))
        _init_time_matrix!(ts, prob.tspan[1])
        us = allocate(backend, typeof(prob.u0), (length(saveat_converted), length(probs)))
    end

    us = adapt(backend, us)
    ts = adapt(backend, ts)
    tstops = adapt(backend, tstops)

    if saveat_converted !== nothing
        saveat_converted = adapt(backend, saveat_converted)
    end
    if backend isa CPU
        @warn "Running the kernel on CPU"
    end

    ode_asolve_kernel(backend)(
        probs, alg, us, ts, _pack_kernel_scalar(backend, dt), callback, tstops,
        _pack_kernel_scalar(backend, abstol), _pack_kernel_scalar(backend, reltol),
        saveat_converted, Val(save_everystep);
        ndrange = length(probs)
    )

    return ts, us
end

function vectorized_asolve(
        probs, prob::SDEProblem, alg;
        dt, saveat = nothing,
        save_everystep = true,
        debug = false,
        kwargs...
    )
    error("Adaptive time-stepping is not supported yet with GPUEM.")
end
