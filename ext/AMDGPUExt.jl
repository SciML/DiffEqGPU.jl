module AMDGPUExt
isdefined(Base, :get_extension) ? (using AMDGPU) : (using ..AMDGPU)
import DiffEqGPU

using .AMDGPU
import .AMDGPU: ROCBackend

DiffEqGPU.maxthreads(::ROCBackend) = 256
DiffEqGPU.maybe_prefer_blocks(::ROCBackend) = ROCBackend()

# Not yet implemented in AMDGPU
# function DiffEqGPU.lufact!(::ROCBackend, W)
#     AMDGPU.rocBLAS.getrf_strided_batched!(W, false)
#     return nothing
# end

end
