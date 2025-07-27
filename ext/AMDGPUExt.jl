module AMDGPUExt
using AMDGPU
import DiffEqGPU

using .AMDGPU
import .AMDGPU: ROCBackend

function DiffEqGPU.EnsembleGPUArray(cpu_offload::Float64)
    DiffEqGPU.EnsembleGPUArray(ROCBackend(), cpu_offload)
end
DiffEqGPU.maxthreads(::ROCBackend) = 256
DiffEqGPU.maybe_prefer_blocks(::ROCBackend) = ROCBackend()

# Not yet implemented in AMDGPU
# function DiffEqGPU.lufact!(::ROCBackend, W)
#     AMDGPU.rocBLAS.getrf_strided_batched!(W, false)
#     return nothing
# end

end
