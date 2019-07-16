# DiffEqGPU

[![Build Status](https://travis-ci.com/ChrisRackauckas/DiffEqGPU.jl.svg?branch=master)](https://travis-ci.com/ChrisRackauckas/DiffEqGPU.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/ChrisRackauckas/DiffEqGPU.jl?svg=true)](https://ci.appveyor.com/project/ChrisRackauckas/DiffEqGPU-jl)
[![Codecov](https://codecov.io/gh/ChrisRackauckas/DiffEqGPU.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ChrisRackauckas/DiffEqGPU.jl)
[![Coveralls](https://coveralls.io/repos/github/ChrisRackauckas/DiffEqGPU.jl/badge.svg?branch=master)](https://coveralls.io/github/ChrisRackauckas/DiffEqGPU.jl?branch=master)

This library is a component package of the DifferentialEquations.jl ecosystem. It includes functionality for making
use of GPUs in the differential equation solvers. 

## Within-Method GPU Parallelism with Direct CuArray Usage

The native Julia libraries, including (but not limited to) OrdinaryDiffEq, StochasticDiffEq, and DelayDiffEq, are
compatible with `u0` being a `CuArray`. When this occurs, all array operations take place on the GPU, including
any implicit solves. This is independent of the DiffEqGPU library. These speedup the solution of a differential
equation which is sufficiently large or expensive.

## Parameter-Parallelism with GPU Ensemble Methods

Parameter-parallel GPU methods are provided for the case where a single solve is too cheap to benefit from
within-method parallelism, but the solution of the same structure (same `f`) is required for very many
different choices of `u0` or `p`. For these cases, DiffEqGPU exports the following ensemble algorithms:

- `EnsembleGPUArray`: Utilizes the CuArray setup to parallelize ODE solves across the GPU. 
- `EnsembleCPUArray`: A test version for analyzing the overhead of the array-based parallelism setup.

For more information on using the ensemble interface, see 
[the DiffEqDocs page on ensembles](http://docs.juliadiffeq.org/latest/features/ensemble.html)
