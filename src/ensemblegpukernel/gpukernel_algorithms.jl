"""
GPUTsit5()

A specialized implementation of the 5th order `Tsit5` method specifically for kernel
generation with EnsembleGPUKernel. For a similar CPU implementation, see
SimpleATsit5 from SimpleDiffEq.jl.
"""
struct GPUTsit5 <: GPUODEAlgorithm end

"""
GPUVern7()

A specialized implementation of the 7th order `Vern7` method specifically for kernel
generation with EnsembleGPUKernel.
"""
struct GPUVern7 <: GPUODEAlgorithm end

"""
GPUVern9()

A specialized implementation of the 9th order `Vern9` method specifically for kernel
generation with EnsembleGPUKernel.
"""
struct GPUVern9 <: GPUODEAlgorithm end

"""
GPURosenbrock23()

A specialized implementation of the W-method `Rosenbrock23` method specifically for kernel
generation with EnsembleGPUKernel.
"""
struct GPURosenbrock23{AD} <: GPUODEImplicitAlgorithm{AD} end

"""
GPURodas4()

A specialized implementation of the `Rodas4` method specifically for kernel
generation with EnsembleGPUKernel.
"""
struct GPURodas4{AD} <: GPUODEImplicitAlgorithm{AD} end

"""
GPURodas5P()

A specialized implementation of the `Rodas5P` method specifically for kernel
generation with EnsembleGPUKernel.
"""
struct GPURodas5P{AD} <: GPUODEImplicitAlgorithm{AD} end

"""
GPUKvaerno3()

A specialized implementation of the `Kvaerno3` method specifically for kernel
generation with EnsembleGPUKernel.
"""
struct GPUKvaerno3{AD} <: GPUODEImplicitAlgorithm{AD} end

"""
GPUKvaerno5()

A specialized implementation of the `Kvaerno5` method specifically for kernel
generation with EnsembleGPUKernel.
"""
struct GPUKvaerno5{AD} <: GPUODEImplicitAlgorithm{AD} end

for Alg in [:GPURosenbrock23, :GPURodas4, :GPURodas5P, :GPUKvaerno3, :GPUKvaerno5]
    @eval begin
        function $Alg(; autodiff = Val{true}())
            $Alg{SciMLBase._unwrap_val(autodiff)}()
        end
    end
end

"""
GPUEM()

A specialized implementation of the Euler-Maruyama `GPUEM` method with weak order 1.0. Made specifically for kernel
generation with EnsembleGPUKernel.
"""
struct GPUEM <: GPUSDEAlgorithm end

"""
GPUSIEA()

A specialized implementation of the weak order 2.0 for Ito SDEs `GPUSIEA` method specifically for kernel
generation with EnsembleGPUKernel.
"""
struct GPUSIEA <: GPUSDEAlgorithm end