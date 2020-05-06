# exponential kernel with total variation distance
struct TVExponentialKernel <: KernelFunctions.SimpleKernel end

# TODO: Not needed in KernelFunctions 1.0
function (kernel::TVExponentialKernel)(x, y)
    return kappa(kernel, evaluate(KernelFunctions.metric(kernel), x, y))
end

KernelFunctions.kappa(kernel::TVExponentialKernel, d) = exp(-d)
KernelFunctions.metric(::TVExponentialKernel) = TotalVariation()

# more efficient application of ScaleTransform
# Optimizations for scale transforms of simple kernels to save allocations:
# Instead of a multiplying every element of the inputs before evaluating the metric,
# we perform a scalar multiplcation of the distance of the original inputs, if possible.
function (k::TransformedKernel{<:TVExponentialKernel,<:ScaleTransform})(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)
    return kappa(k.kernel,
                 KernelFunctions._scale(k.transform,
                                        KernelFunctions.metric(k.kernel), x, y))
end
