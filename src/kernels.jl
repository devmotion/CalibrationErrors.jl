struct TensorProductKernel{K1<:Kernel,K2<:Kernel} <: Kernel{IdentityTransform}
    kernel1::K1
    kernel2::K2
end

KernelFunctions.kappa(kernel::TensorProductKernel, (x1, x2), (y1, y2)) =
    kappa(kernel.kernel1, x1, y1) * kappa(kernel.kernel2, x2, y2)

(kernel::TensorProductKernel)(x, y) = kappa(kernel, x, y)