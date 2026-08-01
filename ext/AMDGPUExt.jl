module AMDGPUExt
using AMDGPU: ROCBackend
import DiffEqGPU

function DiffEqGPU.EnsembleGPUArray(cpu_offload::Float64)
    return DiffEqGPU.EnsembleGPUArray(ROCBackend(), cpu_offload)
end
DiffEqGPU.maxthreads(::ROCBackend) = 256
DiffEqGPU.maybe_prefer_blocks(::ROCBackend) = ROCBackend()

# Not yet implemented in AMDGPU
# function DiffEqGPU.lufact!(::ROCBackend, W)
#     AMDGPU.rocBLAS.getrf_strided_batched!(W, false)
#     return nothing
# end

end
