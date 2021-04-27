@testset "kernels.jl" begin
    # alternative implementation of white kernel
    struct WhiteKernel2 <: Kernel end
    (::WhiteKernel2)(x, y) = x == y

    # alternative implementation TensorProductKernel
    struct TensorProduct2{K1<:Kernel,K2<:Kernel} <: Kernel
        kernel1::K1
        kernel2::K2
    end
    function (kernel::TensorProduct2)((x1, x2), (y1, y2))
        return kernel.kernel1(x1, y1) * kernel.kernel2(x2, y2)
    end

    @testset "TVExponentialKernel" begin
        kernel = TVExponentialKernel()

        # traits
        @test KernelFunctions.metric(kernel) === TotalVariation()

        # simple evaluation
        x, y = rand(10), rand(10)
        @test kernel(x, y) == exp(-totalvariation(x, y))

        # transformations
        @test (kernel ∘ ScaleTransform(0.1))(x, y) == exp(-0.1 * totalvariation(x, y))
        ard = rand(10)
        @test (kernel ∘ ARDTransform(ard))(x, y) == exp(-totalvariation(ard .* x, ard .* y))
    end

    @testset "unsafe_skce_eval" begin
        kernel = SqExponentialKernel()
        kernel1 = kernel ⊗ WhiteKernel()
        kernel2 = kernel ⊗ WhiteKernel2()
        kernel3 = TensorProduct2(kernel, WhiteKernel())

        x1, x2 = rand(10), rand(1:10)

        @test CalibrationErrors.unsafe_skce_eval(kernel1, x1, x2, x1, x2) ≈
              CalibrationErrors.unsafe_skce_eval(kernel2, x1, x2, x1, x2)
        @test CalibrationErrors.unsafe_skce_eval(kernel1, x1, x2, x1, x2) ≈
              CalibrationErrors.unsafe_skce_eval(kernel3, x1, x2, x1, x2)

        y1, y2 = rand(10), rand(1:10)

        @test CalibrationErrors.unsafe_skce_eval(kernel1, x1, x2, y1, y2) ≈
              CalibrationErrors.unsafe_skce_eval(kernel2, x1, x2, y1, y2)
        @test CalibrationErrors.unsafe_skce_eval(kernel1, x1, x2, y1, y2) ≈
              CalibrationErrors.unsafe_skce_eval(kernel3, x1, x2, y1, y2)
    end
end
