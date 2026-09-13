using Enzyme, CUDACore, Test
struct EnzymeAllocationRecord
    value::Float64
    flag::Bool
end

recordalloc(dims) = CuArray{EnzymeAllocationRecord}(undef, dims)

@testset "Allocate records $dims" for dims in ((), (0,), (3,), (2, 3))
    dup = Enzyme.autodiff(ForwardWithPrimal, recordalloc, Duplicated, Const(dims))
    @test size(dup[1]) == dims
    @test all(x -> iszero(x.value) && !x.flag, Array(dup[1]))

    fwd, rev = Enzyme.autodiff_thunk(
        ReverseSplitWithPrimal, Const{typeof(recordalloc)},
        Duplicated, Const{typeof(dims)}
    )
    tape, prim, shad = fwd(Const(recordalloc), Const(dims))
    @test size(shad) == dims
    @test all(x -> iszero(x.value) && !x.flag, Array(shad))
end
