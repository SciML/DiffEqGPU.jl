using DiffEqGPU, Test
using KernelAbstractions: Backend, allocate
using LinearAlgebra: UnitLowerTriangular, UpperTriangular

include("utils.jl")

function portable_lufact!(backend, W)
    return invoke(DiffEqGPU.lufact!, Tuple{Backend, Any}, backend, W)
end

@testset "Batched LU $T, size $n, batches $nbatch" for T in (Float32, Float64),
        n in (1, 3, 5), nbatch in (1, 257)
    A = [T(i == j ? 3n + i + k / 257 : (i - 2j) / (n + k)) for i in 1:n, j in 1:n, k in 1:nbatch]
    for factorize! in (DiffEqGPU.lufact!, portable_lufact!)
        W = allocate(backend, T, n, n, nbatch)
        copyto!(W, A)
        b = reshape(T.(1:(n * nbatch)), n, nbatch)
        x = allocate(backend, T, n, nbatch)
        rhs = allocate(backend, T, n, nbatch)
        copyto!(rhs, b)
        @test factorize!(backend, W) === nothing
        DiffEqGPU.LinSolveGPUSplitFactorize(n, nbatch)(x, W, rhs)
        factors = Array(W)
        for k in 1:nbatch
            F = factors[:, :, k]
            @test UnitLowerTriangular(F) * UpperTriangular(F) ≈ A[:, :, k]
        end
        solutions = Array(x)
        @test all(k -> isapprox(A[:, :, k] * solutions[:, k], b[:, k]), 1:nbatch)
    end
end

@testset "Empty batched LU" begin
    W = allocate(backend, Float32, 3, 3, 0)
    @test portable_lufact!(backend, W) === nothing
    @test size(W) == (3, 3, 0)
end
