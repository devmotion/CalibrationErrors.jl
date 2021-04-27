@testset "ece.jl" begin
    @testset "Trivial tests" begin
        ece = ECE(UniformBinning(10))

        @test @inferred(ece([[0, 1], [1, 0]], [2, 1])) == 0
        @test @inferred(ece([[0, 1], [0.5, 0.5], [0.5, 0.5], [1, 0]], [2, 2, 1, 1])) == 0
    end

    @testset "Uniform binning: Basic properties" begin
        ece = ECE(UniformBinning(10))
        estimates = Vector{Float64}(undef, 1_000)

        for nclasses in (2, 10, 100)
            dist = Dirichlet(nclasses, 1.0)

            predictions = [Vector{Float64}(undef, nclasses) for _ in 1:20]
            targets = Vector{Int}(undef, 20)

            for i in 1:length(estimates)
                rand!.(Ref(dist), predictions)
                targets .= rand.(Categorical.(predictions))

                estimates[i] = ece(predictions, targets)
            end

            @test all(x -> zero(x) < x < one(x), estimates)
        end
    end

    @testset "Median variance binning: Basic properties" begin
        ece = ECE(MedianVarianceBinning(10))
        estimates = Vector{Float64}(undef, 1_000)

        for nclasses in (2, 10, 100)
            dist = Dirichlet(nclasses, 1.0)

            predictions = [Vector{Float64}(undef, nclasses) for _ in 1:20]
            targets = Vector{Int}(undef, 20)

            for i in 1:length(estimates)
                rand!.(Ref(dist), predictions)
                targets .= rand.(Categorical.(predictions))

                estimates[i] = ece(predictions, targets)
            end

            @test all(x -> zero(x) < x < one(x), estimates)
        end
    end
end
