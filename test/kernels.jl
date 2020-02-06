using CalibrationErrors
using CalibrationErrors: skce_kernel
using KernelFunctions

using Random
using Test

Random.seed!(1234)

# alternative implementation of white kernel
struct WhiteKernel2 <: Kernel{IdentityTransform} end
KernelFunctions.kappa(::WhiteKernel2, x, y) = x == y 

# alternative implementation TensorProductKernel
struct TensorProductKernel2{K1<:Kernel,K2<:Kernel} <: Kernel{IdentityTransform}
    kernel1::K1
    kernel2::K2
end
KernelFunctions.kappa(kernel::TensorProductKernel2, (x1, x2), (y1, y2)) =
    kappa(kernel.kernel1, x1, y1) * kappa(kernel.kernel2, x2, y2)

@testset "TensorProductKernel" begin
    kernel = TensorProductKernel(SqExponentialKernel(), ExponentialKernel())

    x1, x2 = rand(10), rand(1:10)
    y1, y2 = rand(10), rand(1:10)

    @test kernel((x1, x2), (y1, y2)) == kappa(kernel.kernel1, x1, y1) * kappa(kernel.kernel2, x2, y2)
end

@testset "skce_kernel" begin
    kernel1 = TensorProductKernel(SqExponentialKernel(2), WhiteKernel())
    kernel2 = TensorProductKernel(SqExponentialKernel(2), WhiteKernel2())
    kernel3 = TensorProductKernel2(SqExponentialKernel(2), WhiteKernel())

    x1, x2 = rand(10), rand(1:10)

    @test skce_kernel(kernel1, x1, x2, x1, x2) ≈ skce_kernel(kernel2, x1, x2, x1, x2)
    @test skce_kernel(kernel1, x1, x2, x1, x2) ≈ skce_kernel(kernel3, x1, x2, x1, x2)

    y1, y2 = rand(10), rand(1:10)

    @test skce_kernel(kernel1, x1, x2, y1, y2) ≈ skce_kernel(kernel2, x1, x2, y1, y2)
    @test skce_kernel(kernel1, x1, x2, y1, y2) ≈ skce_kernel(kernel3, x1, x2, y1, y2)
end