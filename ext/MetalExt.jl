module MetalExt
using Metal
import DiffEqGPU

using .Metal
import .Metal: MetalBackend

DiffEqGPU.maxthreads(::MetalBackend) = 256
DiffEqGPU.maybe_prefer_blocks(::MetalBackend) = MetalBackend()

end
