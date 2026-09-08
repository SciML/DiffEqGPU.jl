# Adaptive controller conventions

Preserve the existing `GPUTsit5` PI-controller behavior. Alternative controllers must be explicit opt-in variants, with documentation explaining their step-size stability and accuracy tradeoffs. Do not change the default based on equal-tolerance timing alone: use work–precision comparisons at matched achieved error, include rejected steps, and test beyond simple nonstiff problems such as Lorenz.
