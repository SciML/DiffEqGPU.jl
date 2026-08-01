module MetalExt
using Metal: MetalBackend
import DiffEqGPU

DiffEqGPU.maxthreads(::MetalBackend) = 256
DiffEqGPU.maybe_prefer_blocks(::MetalBackend) = MetalBackend()

end
