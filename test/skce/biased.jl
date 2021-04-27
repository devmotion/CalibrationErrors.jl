@testset "biased.jl" begin
    @testset "Two-dimensional example" begin
        skce = BiasedSKCE(SqExponentialKernel() ⊗ WhiteKernel())

        # only two predictions, i.e., three unique terms in the estimator
        @test iszero(@inferred(skce([[1, 0], [0, 1]], [1, 2])))
        @test @inferred(skce([[1, 0], [0, 1]], [1, 1])) ≈ 0.5
        @test @inferred(skce([[1, 0], [0, 1]], [2, 1])) ≈ 1 - exp(-1)
        @test @inferred(skce([[1, 0], [0, 1]], [2, 2])) ≈ 0.5
    end

    @testset "Basic properties" begin
        skce = BiasedSKCE((ExponentialKernel() ∘ ScaleTransform(0.1)) ⊗ WhiteKernel())
        estimates = Vector{Float64}(undef, 1_000)

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
    end
end
