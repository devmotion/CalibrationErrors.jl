# exponential kernel with total variation distance
struct TVExponentialKernel <: KernelFunctions.SimpleKernel end

KernelFunctions.kappa(kernel::TVExponentialKernel, d) = exp(-d)
KernelFunctions.metric(::TVExponentialKernel) = TotalVariation()

# Simplifies computations of distances but is type piracy... :(
function KernelFunctions._scale(t::ScaleTransform, metric::TotalVariation, x, y)
    return first(t.s) * evaluate(metric, x, y)
end
