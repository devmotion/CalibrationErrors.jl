using CalibrationErrors
using CalibrationErrors: unsafe_skce_eval
using Distances

using Random
using Test

Random.seed!(1234)

# alternative implementation of white kernel
struct WhiteKernel2 <: Kernel end
(::WhiteKernel2)(x, y) = x == y

@testset "TVExponentialKernel" begin
    kernel = TVExponentialKernel()

    # traits
    @test KernelFunctions.metric(kernel) === TotalVariation()

    # simple evaluation
    x, y = rand(10), rand(10)
    @test kernel(x, y) == exp(-totalvariation(x, y))

    # transformations
    @test transform(kernel, 0.1)(x, y) == exp(- 0.1 * totalvariation(x, y))
    @test transform(kernel, ScaleTransform(0.1))(x, y) ==
        exp(- 0.1 * totalvariation(x, y))
    ard = rand(10)
    @test transform(kernel, ard)(x, y) == exp(- totalvariation(ard .* x, ard .* y))
end

@testset "unsafe_skce_eval" begin
    kernel = transform(SqExponentialKernel(), 2)
    kernel1 = TensorProduct(kernel, WhiteKernel())
    kernel2 = TensorProduct(kernel, WhiteKernel2())
    kernel3 = TensorProduct(kernel, WhiteKernel())

    x1, x2 = rand(10), rand(1:10)

    @test unsafe_skce_eval(kernel1, x1, x2, x1, x2) ≈ unsafe_skce_eval(kernel2, x1, x2, x1, x2)
    @test unsafe_skce_eval(kernel1, x1, x2, x1, x2) ≈ unsafe_skce_eval(kernel3, x1, x2, x1, x2)

    y1, y2 = rand(10), rand(1:10)

    @test unsafe_skce_eval(kernel1, x1, x2, y1, y2) ≈ unsafe_skce_eval(kernel2, x1, x2, y1, y2)
    @test unsafe_skce_eval(kernel1, x1, x2, y1, y2) ≈ unsafe_skce_eval(kernel3, x1, x2, y1, y2)
end
