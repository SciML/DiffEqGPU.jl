function alg_order(alg::Union{GPUODEAlgorithm, GPUSDEAlgorithm})
    error("Order is not defined for this algorithm")
end

alg_order(alg::GPUTsit5) = 5
alg_order(alg::GPUVern7) = 7
alg_order(alg::GPUVern9) = 9
alg_order(alg::GPURosenbrock23) = 2
alg_order(alg::GPURodas4) = 4
alg_order(alg::GPURodas5P) = 5
alg_order(alg::GPUKvaerno3) = 3
alg_order(alg::GPUKvaerno5) = 5

alg_order(alg::GPUEM) = 1
alg_order(alg::GPUSIEA) = 2

function finite_diff_jac(f, jac_prototype, x)
    dx = sqrt(eps(DiffEqBase.RecursiveArrayTools.recursive_bottom_eltype(x)))
    jac = MMatrix{size(x, 1), size(x, 1), eltype(x)}(1I)
    for i in eachindex(x)
        x_dx = convert(MArray, x)
        x_dx[i] = x_dx[i] + dx
        x_dx = convert(SArray, x_dx)
        jac[:, i] .= (f(x_dx) - f(x)) / dx
    end
    return convert(SMatrix, jac)
end

function alg_autodiff(alg::GPUODEAlgorithm)
    error("This algorithm does not have an autodifferentiation option defined.")
end

alg_autodiff(::GPUODEImplicitAlgorithm{AD}) where {AD} = AD
