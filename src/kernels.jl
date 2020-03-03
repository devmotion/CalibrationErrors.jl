## tensor product kernel
struct TensorProductKernel{K1<:Kernel,K2<:Kernel} <: Kernel
    kernel1::K1
    kernel2::K2
end

KernelFunctions.kappa(kernel::TensorProductKernel, (x1, x2), (y1, y2)) =
    kappa(kernel.kernel1, x1, y1) * kappa(kernel.kernel2, x2, y2)

(kernel::TensorProductKernel)(x, y) = kappa(kernel, x, y)

# exponential kernel with total variation distance
struct TVExponentialKernel <: KernelFunctions.BaseKernel end

KernelFunctions.kappa(kernel::TVExponentialKernel, d) = exp(-d) 
KernelFunctions.metric(::TVExponentialKernel) = TotalVariation()
KernelFunctions.iskroncompatible(::TVExponentialKernel) = true

# more efficient application of ScaleTransform
KernelFunctions._scale(t::ScaleTransform, metric::TotalVariation, x, y) = 
    first(t.s) * evaluate(metric, x, y)