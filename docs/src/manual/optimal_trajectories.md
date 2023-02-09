# Choosing Optimal Numbers of Trajectories

There is a balance between two things for choosing the number of trajectories:

  - The number of trajectories needs to be high enough that the work per kernel
    is sufficient to overcome the kernel call cost.
  - More trajectories means that every trajectory will need more time steps, since
    the adaptivity syncs all solves.

From our testing, the balance is found at around 10,000 trajectories being optimal for
EnsembleGPUArray, since it has higher kernel call costs because every internal operation
of the ODE solver requires a kernel call. Thus, for larger sets of trajectories, use a
batch size of 10,000. Of course, benchmark for yourself on your own setup, as all GPUs
are different.

On the other hand, EnsembleGPUKernel fuses the entire GPU solve into a single kernel,
greatly reducing the kernel call cost. This means longer or more expensive ODE solves
will require even less of a percentage of time kernel launching, making the cutoff
much smaller. We see some cases with around 100 ODEs being viable with EnsembleGPUKernel.
Again, this is highly dependent on the ODE and the chosen GPU and thus one will need
to benchmark to get accurate numbers for their system, this is merely a ballpark estimate.
