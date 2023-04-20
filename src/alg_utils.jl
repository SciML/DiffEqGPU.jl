function alg_order(alg::Union{GPUODEAlgorithm, GPUSDEAlgorithm})
    error("Order is not defined for this algorithm")
end

alg_order(alg::GPUTsit5) = 5
alg_order(alg::GPUVern7) = 7
alg_order(alg::GPUVern9) = 9
alg_order(alg::GPURosenbrock23) = 2
alg_order(alg::GPURodas4) = 4

alg_order(alg::GPUEM) = 1
alg_order(alg::GPUSIEA) = 2
