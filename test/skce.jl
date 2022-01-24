@testset "skce.jl" begin
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

    @testset "unsafe_skce_eval" begin
        @testset "binary classification" begin
            # probabilities and boolean targets
            p, p̃ = rand(2)
            y, ỹ = rand(Bool, 2)
            scale = rand()
            kernel = SqExponentialKernel() ∘ ScaleTransform(scale)
            val = unsafe_skce_eval(kernel ⊗ WhiteKernel(), p, y, p̃, ỹ)
            @test unsafe_skce_eval(kernel ⊗ WhiteKernel2(), p, y, p̃, ỹ) ≈ val
            @test unsafe_skce_eval(TensorProduct2(kernel, WhiteKernel()), p, y, p̃, ỹ) ≈
                val
            @test unsafe_skce_eval(TensorProduct2(kernel, WhiteKernel2()), p, y, p̃, ỹ) ≈
                val

            # corresponding values and kernel for full categorical distribution
            pfull = [p, 1 - p]
            yint = 2 - y
            p̃full = [p̃, 1 - p̃]
            ỹint = 2 - ỹ
            kernelfull = SqExponentialKernel() ∘ ScaleTransform(scale / sqrt(2))

            @test unsafe_skce_eval(kernelfull ⊗ WhiteKernel(), pfull, yint, p̃full, ỹint) ≈
                val
            @test unsafe_skce_eval(
                kernelfull ⊗ WhiteKernel2(), pfull, yint, p̃full, ỹint
            ) ≈ val
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
            @test unsafe_skce_eval(TensorProduct2(kernel, WhiteKernel()), p, y, p̃, ỹ) ≈
                val
            @test unsafe_skce_eval(TensorProduct2(kernel, WhiteKernel2()), p, y, p̃, ỹ) ≈
                val
        end
    end

    @testset "Unbiased: Two-dimensional example" begin
        # categorical distributions
        skce = SKCE(SqExponentialKernel() ⊗ WhiteKernel())
        for predictions in ([[1, 0], [0, 1]], ColVecs([1 0; 0 1]), RowVecs([1 0; 0 1]))
            @test iszero(@inferred(skce(predictions, [1, 2])))
            @test iszero(@inferred(skce(predictions, [1, 1])))
            @test @inferred(skce(predictions, [2, 1])) ≈ -2 * exp(-1)
            @test iszero(@inferred(skce(predictions, [2, 2])))
        end

        # probabilities
        skce = SKCE((SqExponentialKernel() ∘ ScaleTransform(sqrt(2))) ⊗ WhiteKernel())
        @test iszero(@inferred(skce([1, 0], [true, false])))
        @test iszero(@inferred(skce([1, 0], [true, true])))
        @test @inferred(skce([1, 0], [false, true])) ≈ -2 * exp(-1)
        @test iszero(@inferred(skce([1, 0], [false, false])))
    end

    @testset "Unbiased: Basic properties" begin
        skce = SKCE((ExponentialKernel() ∘ ScaleTransform(0.1)) ⊗ WhiteKernel())
        estimates = Vector{Float64}(undef, 1_000)

        # categorical distributions
        for nclasses in (2, 10, 100)
            dist = Dirichlet(nclasses, 1.0)

            predictions = [Vector{Float64}(undef, nclasses) for _ in 1:20]
            targets = Vector{Int}(undef, 20)

            for i in 1:length(estimates)
                rand!.(Ref(dist), predictions)
                targets .= rand.(Categorical.(predictions))

                estimates[i] = skce(predictions, targets)
            end

            @test any(x -> x > zero(x), estimates)
            @test any(x -> x < zero(x), estimates)
            @test mean(estimates) ≈ 0 atol = 1e-3
        end

        # probabilities
        predictions = Vector{Float64}(undef, 20)
        targets = Vector{Bool}(undef, 20)
        for i in 1:length(estimates)
            rand!(predictions)
            map!(targets, predictions) do p
                rand() < p
            end
            estimates[i] = skce(predictions, targets)
        end

        @test any(x -> x > zero(x), estimates)
        @test any(x -> x < zero(x), estimates)
        @test mean(estimates) ≈ 0 atol = 1e-3
    end

    @testset "Biased: Two-dimensional example" begin
        # categorical distributions
        skce = SKCE(SqExponentialKernel() ⊗ WhiteKernel(); unbiased=false)
        for predictions in ([[1, 0], [0, 1]], ColVecs([1 0; 0 1]), RowVecs([1 0; 0 1]))
            @test iszero(@inferred(skce(predictions, [1, 2])))
            @test @inferred(skce(predictions, [1, 1])) ≈ 0.5
            @test @inferred(skce(predictions, [2, 1])) ≈ 1 - exp(-1)
            @test @inferred(skce(predictions, [2, 2])) ≈ 0.5
        end

        # probabilities
        skce = SKCE(
            (SqExponentialKernel() ∘ ScaleTransform(sqrt(2))) ⊗ WhiteKernel();
            unbiased=false,
        )
        @test iszero(@inferred(skce([1, 0], [true, false])))
        @test @inferred(skce([1, 0], [true, true])) ≈ 0.5
        @test @inferred(skce([1, 0], [false, true])) ≈ 1 - exp(-1)
        @test @inferred(skce([1, 0], [false, false])) ≈ 0.5
    end

    @testset "Biased: Basic properties" begin
        skce = SKCE(
            (ExponentialKernel() ∘ ScaleTransform(0.1)) ⊗ WhiteKernel(); unbiased=false
        )
        estimates = Vector{Float64}(undef, 1_000)

        # categorical distributions
        for nclasses in (2, 10, 100)
            dist = Dirichlet(nclasses, 1.0)

            predictions = [Vector{Float64}(undef, nclasses) for _ in 1:20]
            targets = Vector{Int}(undef, 20)

            for i in 1:length(estimates)
                rand!.(Ref(dist), predictions)
                targets .= rand.(Categorical.(predictions))
                estimates[i] = skce(predictions, targets)
            end

            @test all(x -> x > zero(x), estimates)
        end

        # probabilities
        predictions = Vector{Float64}(undef, 20)
        targets = Vector{Bool}(undef, 20)
        for i in 1:length(estimates)
            rand!(predictions)
            map!(targets, predictions) do p
                rand() < p
            end
            estimates[i] = skce(predictions, targets)
        end
        @test all(x -> x > zero(x), estimates)
    end

    @testset "Block: Two-dimensional example" begin
        # categorical distributions
        skce = SKCE(SqExponentialKernel() ⊗ WhiteKernel(); blocksize=2)
        for predictions in ([[1, 0], [0, 1]], ColVecs([1 0; 0 1]), RowVecs([1 0; 0 1]))
            @test iszero(@inferred(skce(predictions, [1, 2])))
            @test iszero(@inferred(skce(predictions, [1, 1])))
            @test @inferred(skce(predictions, [2, 1])) ≈ -2 * exp(-1)
            @test iszero(@inferred(skce(predictions, [2, 2])))
        end

        # two predictions, ten times replicated
        for predictions in (
            repeat([[1, 0], [0, 1]], 10),
            ColVecs(repeat([1 0; 0 1], 1, 10)),
            RowVecs(repeat([1 0; 0 1], 10, 1)),
        )
            @test iszero(@inferred(skce(predictions, repeat([1, 2], 10))))
            @test iszero(@inferred(skce(predictions, repeat([1, 1], 10))))
            @test @inferred(skce(predictions, repeat([2, 1], 10))) ≈ -2 * exp(-1)
            @test iszero(@inferred(skce(predictions, repeat([2, 2], 10))))
        end

        # probabilities
        skce = SKCE(
            (SqExponentialKernel() ∘ ScaleTransform(sqrt(2))) ⊗ WhiteKernel(); blocksize=2
        )
        @test iszero(@inferred(skce([1, 0], [true, false])))
        @test iszero(@inferred(skce([1, 0], [true, true])))
        @test @inferred(skce([1, 0], [false, true])) ≈ -2 * exp(-1)
        @test iszero(@inferred(skce([1, 0], [false, false])))

        # two predictions, ten times replicated
        @test iszero(@inferred(skce(repeat([1, 0], 10), repeat([true, false], 10))))
        @test iszero(@inferred(skce(repeat([1, 0], 10), repeat([true, true], 10))))
        @test @inferred(skce(repeat([1, 0], 10), repeat([false, true], 10))) ≈ -2 * exp(-1)
        @test iszero(@inferred(skce(repeat([1, 0], 10), repeat([false, false], 10))))
    end

    @testset "Block: Basic properties" begin
        nsamples = 20
        kernel = (ExponentialKernel() ∘ ScaleTransform(0.1)) ⊗ WhiteKernel()
        skce = SKCE(kernel)
        blockskce = SKCE(kernel; blocksize=2)
        blockskce_all = SKCE(kernel; blocksize=nsamples)
        estimates = Vector{Float64}(undef, 1_000)

        # categorical distributions
        for nclasses in (2, 10, 100)
            dist = Dirichlet(nclasses, 1.0)

            predictions = [Vector{Float64}(undef, nclasses) for _ in 1:nsamples]
            targets = Vector{Int}(undef, nsamples)

            for i in 1:length(estimates)
                rand!.(Ref(dist), predictions)
                targets .= rand.(Categorical.(predictions))

                estimates[i] = blockskce(predictions, targets)

                # consistency checks
                @test estimates[i] ≈ mean(
                    skce(predictions[(2 * i - 1):(2 * i)], targets[(2 * i - 1):(2 * i)]) for
                    i in 1:(nsamples ÷ 2)
                )
                @test skce(predictions, targets) == blockskce_all(predictions, targets)
            end

            @test any(x -> x > zero(x), estimates)
            @test any(x -> x < zero(x), estimates)
            @test mean(estimates) ≈ 0 atol = 5e-3
        end

        # probabilities
        predictions = Vector{Float64}(undef, nsamples)
        targets = Vector{Bool}(undef, nsamples)

        for i in 1:length(estimates)
            rand!(predictions)
            map!(targets, predictions) do p
                return rand() < p
            end
            estimates[i] = blockskce(predictions, targets)

            # consistency checks
            @test estimates[i] ≈ mean(
                skce(predictions[(2 * i - 1):(2 * i)], targets[(2 * i - 1):(2 * i)]) for
                i in 1:(nsamples ÷ 2)
            )
            @test skce(predictions, targets) == blockskce_all(predictions, targets)
        end

        @test any(x -> x > zero(x), estimates)
        @test any(x -> x < zero(x), estimates)
        @test mean(estimates) ≈ 0 atol = 5e-3
    end
end
