@testset "biased.jl" begin
    @testset "Two-dimensional example" begin
        # categorical distributions
        skce = BiasedSKCE(SqExponentialKernel() ⊗ WhiteKernel())
        for predictions in ([[1, 0], [0, 1]], ColVecs([1 0; 0 1]), RowVecs([1 0; 0 1]))
            @test iszero(@inferred(skce(predictions, [1, 2])))
            @test @inferred(skce(predictions, [1, 1])) ≈ 0.5
            @test @inferred(skce(predictions, [2, 1])) ≈ 1 - exp(-1)
            @test @inferred(skce(predictions, [2, 2])) ≈ 0.5
        end

        # probabilities
        skce = BiasedSKCE((SqExponentialKernel() ∘ ScaleTransform(sqrt(2))) ⊗ WhiteKernel())
        @test iszero(@inferred(skce([1, 0], [true, false])))
        @test @inferred(skce([1, 0], [true, true])) ≈ 0.5
        @test @inferred(skce([1, 0], [false, true])) ≈ 1 - exp(-1)
        @test @inferred(skce([1, 0], [false, false])) ≈ 0.5
    end

    @testset "Basic properties" begin
        skce = BiasedSKCE((ExponentialKernel() ∘ ScaleTransform(0.1)) ⊗ WhiteKernel())
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
end
