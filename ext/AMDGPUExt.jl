module AMDGPUExt
using AMDGPU: ROCBackend
import DiffEqGPU

function DiffEqGPU.EnsembleGPUArray(cpu_offload::Float64)
    return DiffEqGPU.EnsembleGPUArray(ROCBackend(), cpu_offload)
end
DiffEqGPU.maxthreads(::ROCBackend) = 256
DiffEqGPU.maybe_prefer_blocks(::ROCBackend) = ROCBackend()

end
