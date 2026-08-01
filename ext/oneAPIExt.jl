module oneAPIExt
using oneAPI: oneAPIBackend
import DiffEqGPU

DiffEqGPU.maxthreads(::oneAPIBackend) = 256
DiffEqGPU.maybe_prefer_blocks(::oneAPIBackend) = oneAPIBackend()

end
