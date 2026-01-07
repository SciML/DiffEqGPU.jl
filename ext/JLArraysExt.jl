module JLArraysExt
using JLArrays: JLBackend
import DiffEqGPU

DiffEqGPU.maxthreads(::JLBackend) = 256
DiffEqGPU.maybe_prefer_blocks(::JLBackend) = JLBackend()

end
