@testset "unbiased.jl" begin
    @testset "Unbiased: Two-dimensional example" begin
        # categorical distributions
        skce = UnbiasedSKCE(SqExponentialKernel() ⊗ WhiteKernel())
        @test iszero(@inferred(skce([[1, 0], [0, 1]], [1, 2])))
        @test iszero(@inferred(skce([[1, 0], [0, 1]], [1, 1])))
        @test @inferred(skce([[1, 0], [0, 1]], [2, 1])) ≈ -2 * exp(-1)
        @test iszero(@inferred(skce([[1, 0], [0, 1]], [2, 2])))

        # probabilities
        skce = UnbiasedSKCE(
            (SqExponentialKernel() ∘ ScaleTransform(sqrt(2))) ⊗ WhiteKernel()
        )
        @test iszero(@inferred(skce([1, 0], [true, false])))
        @test iszero(@inferred(skce([1, 0], [true, true])))
        @test @inferred(skce([1, 0], [false, true])) ≈ -2 * exp(-1)
        @test iszero(@inferred(skce([1, 0], [false, false])))
    end

    @testset "Unbiased: Basic properties" begin
        skce = UnbiasedSKCE((ExponentialKernel() ∘ ScaleTransform(0.1)) ⊗ WhiteKernel())
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

    @testset "Block: Two-dimensional example" begin
        # categorical distributions
        skce = BlockUnbiasedSKCE(SqExponentialKernel() ⊗ WhiteKernel())
        @test iszero(@inferred(skce([[1, 0], [0, 1]], [1, 2])))
        @test iszero(@inferred(skce([[1, 0], [0, 1]], [1, 1])))
        @test @inferred(skce([[1, 0], [0, 1]], [2, 1])) ≈ -2 * exp(-1)
        @test iszero(@inferred(skce([[1, 0], [0, 1]], [2, 2])))

        # two predictions, ten times replicated
        @test iszero(@inferred(skce(repeat([[1, 0], [0, 1]], 10), repeat([1, 2], 10))))
        @test iszero(@inferred(skce(repeat([[1, 0], [0, 1]], 10), repeat([1, 1], 10))))
        @test @inferred(skce(repeat([[1, 0], [0, 1]], 10), repeat([2, 1], 10))) ≈
              -2 * exp(-1)
        @test iszero(@inferred(skce(repeat([[1, 0], [0, 1]], 10), repeat([2, 2], 10))))

        # probabilities
        skce = BlockUnbiasedSKCE(
            (SqExponentialKernel() ∘ ScaleTransform(sqrt(2))) ⊗ WhiteKernel()
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
        skce = UnbiasedSKCE(kernel)
        blockskce = BlockUnbiasedSKCE(kernel)
        blockskce_all = BlockUnbiasedSKCE(kernel, nsamples)
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
