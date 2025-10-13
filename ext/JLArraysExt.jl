module JLArraysExt
using JLArrays: JLBackend
import DiffEqGPU

DiffEqGPU.maybe_prefer_blocks(::JLBackend) = JLBackend()

end
