using CalibrationErrors
using CalibrationErrors: unsafe_skce_eval
using KernelFunctions

using Random
using Test

Random.seed!(1234)

# alternative implementation of white kernel
struct WhiteKernel2 <: Kernel end
KernelFunctions.kappa(::WhiteKernel2, x, y) = x == y 

# alternative implementation TensorProductKernel
struct TensorProductKernel2{K1<:Kernel,K2<:Kernel} <: Kernel
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

@testset "unsafe_skce_eval" begin
    kernel1 = TensorProductKernel(sqexponentialkernel(2), WhiteKernel())
    kernel2 = TensorProductKernel(sqexponentialkernel(2), WhiteKernel2())
    kernel3 = TensorProductKernel2(sqexponentialkernel(2), WhiteKernel())

    x1, x2 = rand(10), rand(1:10)

    @test unsafe_skce_eval(kernel1, x1, x2, x1, x2) ≈ unsafe_skce_eval(kernel2, x1, x2, x1, x2)
    @test unsafe_skce_eval(kernel1, x1, x2, x1, x2) ≈ unsafe_skce_eval(kernel3, x1, x2, x1, x2)

    y1, y2 = rand(10), rand(1:10)

    @test unsafe_skce_eval(kernel1, x1, x2, y1, y2) ≈ unsafe_skce_eval(kernel2, x1, x2, y1, y2)
    @test unsafe_skce_eval(kernel1, x1, x2, y1, y2) ≈ unsafe_skce_eval(kernel3, x1, x2, y1, y2)
end