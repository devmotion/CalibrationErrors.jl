# exponential kernel with total variation distance
struct TVExponentialKernel <: KernelFunctions.SimpleKernel end

KernelFunctions.kappa(kernel::TVExponentialKernel, d) = exp(-d)
KernelFunctions.metric(::TVExponentialKernel) = TotalVariation()
