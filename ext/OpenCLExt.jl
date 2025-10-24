module OpenCLExt
using OpenCL
import DiffEqGPU

using .OpenCL
import .OpenCL: CLBackend

DiffEqGPU.maxthreads(::CLBackend) = 256
DiffEqGPU.maybe_prefer_blocks(::CLBackend) = CLBackend()

end
