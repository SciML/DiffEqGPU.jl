"""
$(DocStringExtensions.README)
"""
module DiffEqGPU

using DocStringExtensions: DocStringExtensions
using KernelAbstractions: KernelAbstractions, @Const, @index, @kernel, CPU
import KernelAbstractions: get_backend, allocate
using SciMLBase: SciMLBase, CallbackSet, CheckInit, ContinuousCallback,
    DiscreteCallback, EnsembleDistributed, EnsembleProblem,
    EnsembleSerial, EnsembleSolution, EnsembleThreads, ODEFunction,
    ODEProblem, ReturnCode, SDEFunction, SDEProblem,
    VectorContinuousCallback, remake, terminate!
using DiffEqBase: DiffEqBase, BrownFullBasicInit
using LinearAlgebra: LinearAlgebra, I, LowerTriangular, NoPivot, RowMaximum,
    SingularException, UpperTriangular, det
using Distributed: Distributed, nprocs, pmap
using ForwardDiff: ForwardDiff
import ChainRulesCore
import ChainRulesCore: NoTangent
using RecursiveArrayTools: RecursiveArrayTools, VectorOfArray
import ZygoteRules
import Base.Threads
using Base: setindex
using CommonSolve: solve
using LinearSolve: LinearSolve
using SimpleNonlinearSolve: SimpleNonlinearSolve
import SimpleNonlinearSolve: SimpleTrustRegion
#For gpu_tsit5
using Adapt: Adapt, adapt
using SimpleDiffEq: SimpleDiffEq, GPUSimpleATsit5, GPUSimpleAVern7, GPUSimpleAVern9,
    GPUSimpleTsit5, GPUSimpleVern7, GPUSimpleVern9, SimpleEM
using StaticArrays: StaticArrays
using StaticArraysCore: MArray, MMatrix, SArray, SMatrix, SVector, Size,
    StaticMatrix, StaticVector, similar_type
using Parameters: Parameters
using MuladdMacro: MuladdMacro, @muladd
using Random: Random
using Setfield: Setfield, @set, @set!
using UnPack: @unpack
# StaticArraysCore-owned type alias (re-exported by StaticArrays); used in dispatch.
import StaticArrays: StaticVecOrMat
# Non-public StaticArrays internals used by the vendored GPU LU/linsolve kernels.
import StaticArrays: @_inline_meta, LU, StaticLUMatrix
import SciMLBase: ImmutableODEProblem

"""
    EnsembleArrayAlgorithm <: SciMLBase.EnsembleAlgorithm

Developer interface for ensemble algorithms that fuse a SciML ensemble into array-valued
state and parameter problems before delegating each trajectory to an ordinary SciML
differential equation solver.

# Interface Rules

Subtypes are used as the third positional argument to `solve(ensembleprob, alg,
ensemblealg; trajectories, kwargs...)` for `SciMLBase.AbstractEnsembleProblem`s. A subtype
must be accepted by DiffEqGPU's `SciMLBase.__solve` method and by the lower-level
`vectorized_map_solve` path. It must define enough backend/device information for
`vectorized_map_solve_up` to move the generated `u0` and `p` arrays to the execution
backend, and it must preserve the SciML ensemble semantics for `prob_func`, `reduction`,
`batch_size`, `trajectories`, and solver keyword arguments.

# Implementations

  - `EnsembleGPUArray`: runs the fused array problem on a KernelAbstractions backend.
  - `EnsembleCPUArray`: keeps the same fused-array code path on CPU for debugging.

# Examples

```julia
solve(ensemble_prob, Tsit5(), EnsembleGPUArray(backend); trajectories = 10_000)
```
"""
abstract type EnsembleArrayAlgorithm <: SciMLBase.EnsembleAlgorithm end

"""
    EnsembleKernelAlgorithm <: SciMLBase.EnsembleAlgorithm

Developer interface for ensemble algorithms that generate one GPU kernel for a complete
fixed-size ODE or SDE solve.

# Interface Rules

Subtypes are used as the ensemble algorithm in `solve(ensembleprob, gpu_alg, ensemblealg;
trajectories, kwargs...)`, where `gpu_alg` is a `GPUODEAlgorithm` or `GPUSDEAlgorithm`.
The corresponding problem must be convertible to the kernel path with
`make_prob_compatible`; in practice this means out-of-place dynamics over static state
containers for `EnsembleGPUKernel`. Implementations must support the lower-level
`vectorized_solve` and, for ODE algorithms, `vectorized_asolve` entry points used by
`batch_solve_up_kernel`.

# Implementations

  - `EnsembleGPUKernel`: compiles a KernelAbstractions kernel for all trajectories in a
    batch.

# Examples

```julia
solve(ensemble_prob, GPUTsit5(), EnsembleGPUKernel(backend);
    trajectories = 10_000, adaptive = false, dt = 0.1f0)
```
"""
abstract type EnsembleKernelAlgorithm <: SciMLBase.EnsembleAlgorithm end

##Solvers for EnsembleGPUKernel
"""
    GPUODEAlgorithm <: SciMLBase.AbstractODEAlgorithm

Developer interface for ODE algorithms supported by `EnsembleGPUKernel`.

# Interface Rules

Subtypes must be immutable algorithm selectors with all per-solve state allocated by the
kernel integrator constructors. They are passed as the second positional argument to
`solve(ensembleprob, alg, EnsembleGPUKernel(backend); kwargs...)` and to the lower-level
`vectorized_solve`/`vectorized_asolve` functions. A subtype must have kernel integrator
support in `batch_solve_up_kernel`, an order from `alg_order`, and any required tableau,
interpolation, callback, nonlinear-solve, or mass-matrix support implemented in GPU-safe
code.

The ODE function must be GPU compilable. The kernel path is tested through the generic
lower-level API by constructing compatible `ODEProblem`s and calling `vectorized_solve`
and `vectorized_asolve` without reaching into solver internals.

# Examples

```julia
DiffEqGPU.vectorized_solve(gpu_probs, prob, GPUTsit5(); dt = 0.1f0)
```
"""
abstract type GPUODEAlgorithm <: SciMLBase.AbstractODEAlgorithm end

"""
    GPUSDEAlgorithm <: SciMLBase.AbstractSDEAlgorithm

Developer interface for SDE algorithms supported by `EnsembleGPUKernel`.

# Interface Rules

Subtypes are passed as the SDE algorithm in `solve(ensembleprob, alg,
EnsembleGPUKernel(backend); kwargs...)` and through the lower-level `vectorized_solve`
entry point. Implementations must provide GPU-safe stepping code, static state support,
and noise compatibility checks before launching kernels. Current kernel SDE algorithms are
fixed-step methods and must reject unsupported noise structures rather than falling back to
host-side behavior.

# Examples

```julia
DiffEqGPU.vectorized_solve(gpu_probs, sde_prob, GPUEM();
    dt = 0.01f0, save_everystep = false)
```
"""
abstract type GPUSDEAlgorithm <: SciMLBase.AbstractSDEAlgorithm end

"""
    GPUODEImplicitAlgorithm{AD} <: GPUODEAlgorithm

Developer interface for stiff ODE algorithms in the `EnsembleGPUKernel` path.

# Type Parameters

  - `AD`: Boolean-like type parameter indicating whether the algorithm may derive missing
    Jacobian and time-gradient information with automatic differentiation. Constructors
    accept `autodiff = Val{true}()` or `Val{false}()`.

# Interface Rules

Subtypes must implement the `GPUODEAlgorithm` rules and additionally provide GPU-safe
linear/nonlinear solve support. They must either receive analytical Jacobian/time-gradient
functions through the `ODEFunction` or use the `AD` parameter to select automatic or finite
difference derivative construction. Their nonlinear solve path is expected to use
`AbstractNLSolver` state built by `build_nlsolver` and to solve static linear systems with
DiffEqGPU's GPU-compatible linear algebra utilities.

# Examples

```julia
solve(ensemble_prob, GPURodas4(autodiff = Val{false}()),
    EnsembleGPUKernel(backend); trajectories = 10_000)
```
"""
abstract type GPUODEImplicitAlgorithm{AD} <: GPUODEAlgorithm end

_unwrap_val(B) = B
_unwrap_val(::Val{B}) where {B} = B

include("ensemblegpuarray/callbacks.jl")
include("ensemblegpuarray/kernels.jl")
include("ensemblegpuarray/problem_generation.jl")
include("ensemblegpuarray/lowerlevel_solve.jl")

include("ensemblegpukernel/callbacks.jl")
include("ensemblegpukernel/lowerlevel_solve.jl")
include("ensemblegpukernel/gpukernel_algorithms.jl")
include("ensemblegpukernel/linalg/lu.jl")
include("ensemblegpukernel/linalg/linsolve.jl")
include("ensemblegpukernel/alg_utils.jl")
include("ensemblegpukernel/integrators/nonstiff/types.jl")
include("ensemblegpukernel/integrators/stiff/types.jl")
include("ensemblegpukernel/integrators/integrator_utils.jl")
include("ensemblegpukernel/integrators/stiff/interpolants.jl")
include("ensemblegpukernel/integrators/nonstiff/interpolants.jl")
include("ensemblegpukernel/nlsolve/type.jl")
include("ensemblegpukernel/nlsolve/utils.jl")
include("ensemblegpukernel/nlsolve/initialization.jl")
include("ensemblegpukernel/kernels.jl")

include("ensemblegpukernel/perform_step/gpu_tsit5_perform_step.jl")
include("ensemblegpukernel/perform_step/gpu_vern7_perform_step.jl")
include("ensemblegpukernel/perform_step/gpu_vern9_perform_step.jl")
include("ensemblegpukernel/perform_step/gpu_em_perform_step.jl")
include("ensemblegpukernel/perform_step/gpu_siea_perform_step.jl")
include("ensemblegpukernel/perform_step/gpu_rosenbrock23_perform_step.jl")
include("ensemblegpukernel/perform_step/gpu_rodas4_perform_step.jl")
include("ensemblegpukernel/perform_step/gpu_rodas5P_perform_step.jl")
include("ensemblegpukernel/perform_step/gpu_kvaerno3_perform_step.jl")
include("ensemblegpukernel/perform_step/gpu_kvaerno5_perform_step.jl")

include("ensemblegpukernel/tableaus/verner_tableaus.jl")
include("ensemblegpukernel/tableaus/rodas_tableaus.jl")
include("ensemblegpukernel/tableaus/kvaerno_tableaus.jl")

include("utils.jl")
include("algorithms.jl")
include("solve.jl")
export EnsembleProblem, EnsembleSolution, EnsembleSerial, EnsembleThreads,
    EnsembleDistributed

export BrownFullBasicInit, CheckInit

export EnsembleCPUArray, EnsembleGPUArray, EnsembleGPUKernel, LinSolveGPUSplitFactorize

export GPUTsit5, GPUVern7, GPUVern9, GPUEM, GPUSIEA
## Stiff ODE solvers
export GPURosenbrock23, GPURodas4, GPURodas5P, GPUKvaerno3, GPUKvaerno5

export terminate!

end # module
