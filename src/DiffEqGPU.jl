"""
$(DocStringExtensions.README)
"""
module DiffEqGPU

using DocStringExtensions
using KernelAbstractions
import KernelAbstractions: get_backend, allocate
using SciMLBase, DiffEqBase, LinearAlgebra, Distributed
using ForwardDiff
import ChainRulesCore
import ChainRulesCore: NoTangent
using RecursiveArrayTools
import ZygoteRules
import Base.Threads
using LinearSolve
#For gpu_tsit5
using Adapt, SimpleDiffEq, StaticArrays
using Parameters, MuladdMacro
using Random
using Setfield
using ForwardDiff
import StaticArrays: StaticVecOrMat, @_inline_meta
# import LinearAlgebra: \
import StaticArrays: LU, StaticLUMatrix, arithmetic_closure

abstract type EnsembleArrayAlgorithm <: SciMLBase.EnsembleAlgorithm end
abstract type EnsembleKernelAlgorithm <: SciMLBase.EnsembleAlgorithm end

##Solvers for EnsembleGPUKernel
abstract type GPUODEAlgorithm <: DiffEqBase.AbstractODEAlgorithm end
abstract type GPUSDEAlgorithm <: DiffEqBase.AbstractSDEAlgorithm end
abstract type GPUODEImplicitAlgorithm{AD} <: GPUODEAlgorithm end

include("ensemblegpuarray/callbacks.jl")
include("ensemblegpuarray/kernels.jl")
include("ensemblegpuarray/problem_generation.jl")
include("ensemblegpuarray/lowerlevel_solve.jl")

include("ensemblegpukernel/problems/ode_problems.jl")
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

export EnsembleCPUArray, EnsembleGPUArray, EnsembleGPUKernel, LinSolveGPUSplitFactorize

export GPUTsit5, GPUVern7, GPUVern9, GPUEM, GPUSIEA
## Stiff ODE solvers
export GPURosenbrock23, GPURodas4, GPURodas5P, GPUKvaerno3, GPUKvaerno5

export terminate!

# This symbol is only defined on Julia versions that support extensions
if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("../ext/CUDAExt.jl")
        @require AMDGPU="21141c5a-9bdb-4563-92ae-f87d6854732e" include("../ext/AMDGPUExt.jl")
        @require oneAPI="8f75cd03-7ff8-4ecb-9b8f-daf728133b1b" include("../ext/oneAPIExt.jl")
        @require Metal="dde4c033-4e86-420c-a63e-0dd931031962" include("../ext/MetalExt.jl")
    end
end

end # module
