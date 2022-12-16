# Choosing the Ensemble: EnsembleGPUArray vs EnsembleGPUKernel

The short answer for how to choose an ensembler is that, if `EnsembleGPUKernel` works on
your problem, you should use it. A more complex discussion is the following:

- `EnsembleGPUKernel` is more asynchronous and has lower kernel call counts than
  `EnsembleGPUArray`. This should amount to lower overhead in any case where the algorithms
  are the same.
- `EnsembleGPUKernel` is restrictive on the types of ODE solvers that have been implemented
  for it. If the most efficient method is not in the list of GPU kernel methods, it may be
  more efficient to use `EnsembleGPUArray` with the better method.
- `EnsembleGPUKernel` requires equations to be written in out-of-place form, along with a
  few other restrictions, and thus in some cases can be less automatic than
  `EnsembleGPUArray` depending on how the code was originally written.
- Currently, `EnsembleGPUKernel` does not support methods for stiff equations.
