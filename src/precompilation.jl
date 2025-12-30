using PrecompileTools

@setup_workload begin
    using StaticArrays
    using ForwardDiff: Dual, value

    @compile_workload begin
        # Precompile algorithm constructors
        GPUTsit5()
        GPUVern7()
        GPUVern9()
        GPURosenbrock23()
        GPURodas4()
        GPURodas5P()
        GPUKvaerno3()
        GPUKvaerno5()
        GPUEM()
        GPUSIEA()

        # Precompile ensemble algorithms
        EnsembleCPUArray()

        # Precompile utility functions with common types
        x_f64 = [1.0, 2.0, 3.0]
        x_f32 = [1.0f0, 2.0f0, 3.0f0]
        diffeqgpunorm(x_f64, 0.0)
        diffeqgpunorm(x_f32, 0.0f0)
        diffeqgpunorm(1.0, 0.0)
        diffeqgpunorm(1.0f0, 0.0f0)

        # Precompile with ForwardDiff Dual numbers
        dual_arr = [Dual(1.0, 1.0), Dual(2.0, 0.0), Dual(3.0, 0.0)]
        diffeqgpunorm(dual_arr, 0.0)
        diffeqgpunorm(Dual(1.0, 1.0), 0.0)

        # Precompile make_prob_compatible
        make_prob_compatible(nothing)
    end
end
