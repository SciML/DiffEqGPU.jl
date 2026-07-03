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

abstract type EnsembleArrayAlgorithm <: SciMLBase.EnsembleAlgorithm end
abstract type EnsembleKernelAlgorithm <: SciMLBase.EnsembleAlgorithm end

##Solvers for EnsembleGPUKernel
abstract type GPUODEAlgorithm <: SciMLBase.AbstractODEAlgorithm end
abstract type GPUSDEAlgorithm <: SciMLBase.AbstractSDEAlgorithm end
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
