module OpenCLExt
using OpenCL
import DiffEqGPU

using .OpenCL
import .OpenCL: OpenCLBackend

DiffEqGPU.maxthreads(::OpenCLBackend) = 256
DiffEqGPU.maybe_prefer_blocks(::OpenCLBackend) = OpenCLBackend()

end
