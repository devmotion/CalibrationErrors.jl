@testset "ece.jl" begin
    @testset "Trivial tests" begin
        ece = ECE(UniformBinning(10))
        for predictions in ([[0, 1], [1, 0]], ColVecs([0 1; 1 0]), RowVecs([0 1; 1 0]))
            @test iszero(@inferred(ece(predictions, [2, 1])))
        end
        for predictions in (
            [[0, 1], [0.5, 0.5], [0.5, 0.5], [1, 0]],
            ColVecs([0 0.5 0.5 1; 1 0.5 0.5 0]),
            RowVecs([0 1; 0.5 0.5; 0.5 0.5; 1 0]),
        )
            @test iszero(@inferred(ece(predictions, [2, 2, 1, 1])))
        end
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
