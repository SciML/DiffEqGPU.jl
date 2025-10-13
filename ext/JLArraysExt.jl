module JLArraysExt
using JLArrays
import DiffEqGPU

using .JLArrays
import .JLArrays: JLBackend

DiffEqGPU.maybe_prefer_blocks(::JLBackend) = JLBackend()

end
