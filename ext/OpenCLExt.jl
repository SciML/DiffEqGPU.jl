module OpenCLExt
using OpenCL: OpenCLBackend
import DiffEqGPU

DiffEqGPU.maxthreads(::OpenCLBackend) = 256
DiffEqGPU.maybe_prefer_blocks(::OpenCLBackend) = OpenCLBackend()

end
