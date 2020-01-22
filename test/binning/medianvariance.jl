using CalibrationErrors, Distributions, StatsBase
using CalibrationErrors: perform

using Random
using Test

Random.seed!(1234)

@testset "Simple example ($nclasses classes)" for nclasses in (2, 10, 100)
    nsamples = 1_000
    dist = Dirichlet(nclasses, 1)
    predictions = [rand(dist) for _ in 1:nsamples]
    targets = rand(1:nclasses, nsamples)

    # set minimum number of samples
    bins = @inferred(perform(MedianVarianceBinning(10), predictions, targets))
    @test all(bin -> bin.nsamples ≥ 10, bins)
    @test sum(bin -> bin.nsamples, bins) == nsamples
    @test sum(bin -> bin.sum_predictions, bins) ≈ sum(predictions)
    @test sum(bin -> bin.counts_targets, bins) == counts(targets, 1:nclasses)

    # set maximum number of bins
    bins = @inferred(perform(MedianVarianceBinning(0, 10), predictions, targets))
    @test length(bins) ≤ 10
    @test sum(bin -> bin.nsamples, bins) == nsamples
    @test sum(bin -> bin.sum_predictions, bins) ≈ sum(predictions)
    @test sum(bin -> bin.counts_targets, bins) == counts(targets, 1:nclasses)
end
