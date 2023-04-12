module oneAPIExt
isdefined(Base, :get_extension) ? (using oneAPI) : (using ..oneAPI)
import DiffEqGPU

using .oneAPI
import .oneAPI: oneAPIBackend

DiffEqGPU.maxthreads(::oneAPIBackend) = 256
DiffEqGPU.maybe_prefer_blocks(::oneAPIBackend) = oneAPIBackend()

end
