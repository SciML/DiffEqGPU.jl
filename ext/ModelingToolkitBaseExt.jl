module ModelingToolkitBaseExt

using ModelingToolkitBase: MTKParameters, System, unknowns
import DiffEqGPU
import SciMLBase

const MTKPARAMETERS_PORTIONS = (
    :tunable, :initials, :discrete, :constant, :nonnumeric, :caches,
)

function DiffEqGPU.make_parameter_compatible(p::MTKParameters)
    compatible = MTKParameters(
        DiffEqGPU.make_static_storage(p.tunable),
        DiffEqGPU.make_static_storage(p.initials),
        DiffEqGPU.make_static_storage(p.discrete),
        DiffEqGPU.make_static_storage(p.constant),
        DiffEqGPU.make_static_storage(p.nonnumeric),
        DiffEqGPU.make_static_storage(p.caches)
    )
    # Converting the storage to `SArray` cannot rescue content that is not isbits in the
    # first place — a nonnumeric buffer holding a type or a closure over an array, say.
    # Such a `p` cannot be a field of the isbits problem uploaded to the device, so say so
    # here rather than failing somewhere inside the kernel.
    isbits(compatible) && return compatible
    offenders = filter(MTKPARAMETERS_PORTIONS) do portion
        !isbits(getproperty(compatible, portion))
    end
    throw(
        ArgumentError(
            "These `MTKParameters` cannot be used by EnsembleGPUKernel: the $(join(offenders, ", ", " and ")) $(length(offenders) == 1 ? "portion holds" : "portions hold") values that are not isbits, so the problem cannot be uploaded to the device. Either keep such values out of the parameter set, or give their type an isbits stand-in by adding a `DiffEqGPU.make_static_storage` method for it."
        )
    )
end

function DiffEqGPU.lower_initialization_problem(prob::SciMLBase.SCCNonlinearProblem)
    sys = prob.f.sys
    sys isa System || throw(
        ArgumentError(
            "Only ModelingToolkit-generated SCC nonlinear initialization problems can be lowered for EnsembleGPUKernel."
        )
    )
    any(p -> nameof(typeof(p)) === :HomotopyProblem, prob.probs) && throw(
        ArgumentError(
            "SCC nonlinear initialization problems containing homotopy blocks are not supported by EnsembleGPUKernel. Recompile the system with `mtkcompile(sys; homotopy = false)` to replace every `homotopy(actual, simplified)` node by `actual`, which builds an equivalent initialization without homotopy blocks."
        )
    )

    block_states = map(prob.probs) do block_prob
        block_u0 = SciMLBase.state_values(block_prob)
        block_u0 !== nothing && return block_u0
        # A linear block is solved by one exact Newton step, which lands on the solution
        # from any seed, so a zero seed stands in for the missing state.
        block_prob isa SciMLBase.LinearProblem || throw(
            ArgumentError(
                "Every nonlinear SCC initialization block must have an initial state for EnsembleGPUKernel."
            )
        )
        zero(block_prob.b)
    end
    u0 = reduce(vcat, block_states)
    length(u0) == length(unknowns(sys)) || throw(
        ArgumentError("SCC initialization block sizes do not match the full state size.")
    )
    f = SciMLBase.NonlinearFunction{false, SciMLBase.FullSpecialize}(
        sys; u0, p = prob.p, check_compatibility = false
    )
    nonlinear_prob = SciMLBase.NonlinearProblem{false}(f, u0, prob.p)

    offset = 0
    blocks = map(prob.probs, block_states) do block_prob, block_u0
        n = length(block_u0)
        block = DiffEqGPU.ImmutableSCCBlock{
            offset + 1, n, block_prob isa SciMLBase.LinearProblem,
        }()
        offset += n
        block
    end
    return DiffEqGPU.ImmutableSCCNonlinearProblem(nonlinear_prob, Tuple(blocks))
end

# ModelingToolkit emits the initialization maps as isbits `RuntimeGeneratedFunction`s
# under `SciMLBase.FullSpecialize` (ModelingToolkit.jl#5043). Those rebuild `u0` and `p`
# in the buffer types of whatever value provider they are handed, so against the static
# storage `make_prob_compatible` installs they produce static results and run unchanged
# inside the kernel. Nothing is lowered here.
device_compatible_map(::Nothing) = true
device_compatible_map(map) = isbitstype(typeof(map))

# The maps read `state_values`/`parameter_values` off whatever they are handed. Inside the
# kernel that is the nonlinear solution; here it is the lowered initialization problem, and
# the SCC wrapper is not itself a value provider.
map_value_provider(prob) = prob
map_value_provider(prob::DiffEqGPU.ImmutableSCCNonlinearProblem) = prob.problem

function DiffEqGPU.make_initialization_maps_compatible(
        prob, initprob, umap, pmap, ::MTKParameters
    )
    device_compatible_map(umap) && device_compatible_map(pmap) || throw(
        ArgumentError(
            "EnsembleGPUKernel requires ModelingToolkit's device-compatible initialization maps, which are emitted only under `SciMLBase.FullSpecialize`. Rebuild the problem as `ODEProblem{iip, SciMLBase.FullSpecialize}(sys, ...)`; ModelingToolkit ignores a `specialize` keyword argument."
        )
    )
    # The maps are evaluated inside the kernel, which cannot allocate, so the `u0` they
    # build has to be static too. ModelingToolkit fixes that container at
    # code-generation time from `u0_constructor`, so it cannot be repaired here.
    umap === nothing || isbits(umap(map_value_provider(initprob))) || throw(
        ArgumentError(
            "ModelingToolkit's state initialization map builds a `$(typeof(umap(map_value_provider(initprob))))`, which a kernel can neither allocate nor hold. It has to build a static, immutable `u0` instead, and that is decided when the problem is constructed. Build the problem out-of-place and with static storage: `ODEProblem{false, SciMLBase.FullSpecialize}(sys, ...; u0_constructor = static_constructor, p_constructor = static_constructor)` where `static_constructor(values) = SVector{length(values)}(values)`. Out-of-place is required as well as static: an `MVector` is a mutable struct and so is not isbits, while an in-place problem cannot write into an immutable `SVector`."
        )
    )
    return umap, pmap
end

end
