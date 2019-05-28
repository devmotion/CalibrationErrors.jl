using CalibrationErrors, Distributions, StatsBase
using CalibrationErrors: perform, binid

using Random

Random.seed!(1234)

@testset "Binning indices" begin
    @testset "One dimension" begin
        @test_throws ArgumentError binid(UniformBinning(10), -0.5)
        @test binid(UniformBinning(10), 0) == 1
        @test binid(UniformBinning(10), 0.1) == 1
        @test binid(UniformBinning(10), 1) == 10
        @test_throws ArgumentError binid(UniformBinning(10), 1.5)
    end

    @testset "Two dimensions" begin
        @test_throws ArgumentError binid(UniformBinning(10), [-0.5, 0.5])
        @test binid(UniformBinning(10), [0, 0]) == 1
        @test binid(UniformBinning(10), [0.1, 0]) == 1
        @test binid(UniformBinning(10), [1, 1]) == 100
        @test_throws ArgumentError binid(UniformBinning(10), [1.5, 0.5])
    end
end

@testset "Simple example ($nclasses classes)" for nclasses in (2, 10, 100)
    # sample predictions and labels
    nsamples = 1_000
    predictions = rand(Dirichlet(nclasses, 1), nsamples)
    labels = rand(1:nclasses, nsamples)

    # bin data in bins of uniform width
    bins = perform(UniformBinning(10), predictions, labels)

    # check all bins
    for (idx, bin) in bins
        idxs = Int[]
        for i in axes(predictions, 2)
            idx == prod(max(1, ceil(10 * p)) for p in view(predictions, :, i)) &&
                push!(idxs, i)
        end

        @test bin.nsamples[] == length(idxs)
        @test bin.prediction_sum ≈ vec(sum(predictions[:, idxs], dims=2))
        @test bin.label_counts ≈ counts(labels[idxs], 1:nclasses)
    end
end
