@testset "generic.jl" begin
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

    @testset "binary classification" begin
        # probabilities and boolean targets
        p, p̃ = rand(2)
        y, ỹ = rand(Bool, 2)
        scale = rand()
        kernel = SqExponentialKernel() ∘ ScaleTransform(scale)
        val = unsafe_skce_eval(kernel ⊗ WhiteKernel(), p, y, p̃, ỹ)
        @test unsafe_skce_eval(kernel ⊗ WhiteKernel2(), p, y, p̃, ỹ) ≈ val
        @test unsafe_skce_eval(TensorProduct2(kernel, WhiteKernel()), p, y, p̃, ỹ) ≈ val
        @test unsafe_skce_eval(TensorProduct2(kernel, WhiteKernel2()), p, y, p̃, ỹ) ≈ val

        # corresponding values and kernel for full categorical distribution
        pfull = [p, 1 - p]
        yint = 2 - y
        p̃full = [p̃, 1 - p̃]
        ỹint = 2 - ỹ
        kernelfull = SqExponentialKernel() ∘ ScaleTransform(scale / sqrt(2))

        @test unsafe_skce_eval(kernelfull ⊗ WhiteKernel(), pfull, yint, p̃full, ỹint) ≈ val
        @test unsafe_skce_eval(kernelfull ⊗ WhiteKernel2(), pfull, yint, p̃full, ỹint) ≈
            val
        @test unsafe_skce_eval(
            TensorProduct2(kernelfull, WhiteKernel()), pfull, yint, p̃full, ỹint
        ) ≈ val
        @test unsafe_skce_eval(
            TensorProduct2(kernelfull, WhiteKernel2()), pfull, yint, p̃full, ỹint
        ) ≈ val
    end

    @testset "multi-class classification" begin
        n = 10
        p = rand(n)
        p ./= sum(p)
        y = rand(1:n)
        p̃ = rand(n)
        p̃ ./= sum(p̃)
        ỹ = rand(1:n)

        kernel = SqExponentialKernel() ∘ ScaleTransform(rand())
        val = unsafe_skce_eval(kernel ⊗ WhiteKernel(), p, y, p̃, ỹ)

        @test unsafe_skce_eval(kernel ⊗ WhiteKernel2(), p, y, p̃, ỹ) ≈ val
        @test unsafe_skce_eval(TensorProduct2(kernel, WhiteKernel()), p, y, p̃, ỹ) ≈ val
        @test unsafe_skce_eval(TensorProduct2(kernel, WhiteKernel2()), p, y, p̃, ỹ) ≈ val
    end
end
