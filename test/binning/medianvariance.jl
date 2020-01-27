using CalibrationErrors, Distributions, StatsBase
using CalibrationErrors: perform

using LinearAlgebra
using Random
using Test

Random.seed!(1234)

@testset "Constructors" begin
    @test_throws ErrorException MedianVarianceBinning(-1)
    @test_throws ErrorException MedianVarianceBinning(0)
    @test_throws ErrorException MedianVarianceBinning(10, -1)
    @test_throws ErrorException MedianVarianceBinning(10, 0)
end

@testset "Basic tests ($nclasses classes)" for nclasses in (2, 10, 100)
    nsamples = 1_000
    dist = Dirichlet(nclasses, 1)
    predictions = [rand(dist) for _ in 1:nsamples]
    targets = rand(1:nclasses, nsamples)

    # set minimum number of samples
    for minsize in (1, 10, 100, 500, 1_000)
        bins = @inferred(perform(MedianVarianceBinning(minsize), predictions, targets))
        @test all(bin -> bin.nsamples ≥ minsize, bins)
        @test sum(bin -> bin.nsamples, bins) == nsamples
        @test sum(bin -> bin.sum_predictions, bins) ≈ sum(predictions)
        @test sum(bin -> bin.counts_targets, bins) == counts(targets, 1:nclasses)
    end

    # set maximum number of bins
    for maxbins in (1, 10, 100, 500, 1_000)
        bins = @inferred(perform(MedianVarianceBinning(1, maxbins), predictions, targets))
        @test length(bins) ≤ maxbins
        @test all(bin -> bin.nsamples ≥ 1, bins)
        @test sum(bin -> bin.nsamples, bins) == nsamples
        @test sum(bin -> bin.sum_predictions, bins) ≈ sum(predictions)
        @test sum(bin -> bin.counts_targets, bins) == counts(targets, 1:nclasses)
    end
end

@testset "Simple example" begin
    predictions = [[0.4, 0.1, 0.5], [0.5, 0.3, 0.2], [0.3, 0.7, 0.0]]
    targets = [1, 2, 3]
    # maximum possible steps in the order they might occur:
    # first step: [1, 2], [3] -> create bin with [3]
    # second step: [2], [1] -> create bins with [2] and [1]

    bins = perform(MedianVarianceBinning(1), predictions, targets)
    @test length(bins) == 3
    @test all(bin -> bin.nsamples == 1, bins)
    for (i, idx) in enumerate((3, 2, 1))
        @test bins[i].sum_predictions == predictions[idx]
        @test bins[i].counts_targets == Matrix{Float64}(I, 3, 3)[:, targets[idx]]
    end

    bins = perform(MedianVarianceBinning(2), predictions, targets)
    @test length(bins) == 1
    @test bins[1].nsamples == 3
    @test bins[1].sum_predictions == sum(predictions)
    @test bins[1].counts_targets == [1, 1, 1]

    predictions = [[0.4, 0.1, 0.5], [0.5, 0.3, 0.2], [0.3, 0.7, 0.0], [0.1, 0.0, 0.9], [0.8, 0.1, 0.1]]
    targets = [1, 2, 3, 1, 2]
    # maximum possible steps in the order they might occur:
    # first step: [2, 3, 5], [1, 4]
    # second step: [2, 5], [3], [1, 4] -> create bin with [3]
    # third step:  [2, 5], [1], [4] -> create bins with [1] and [4]
    # fourth step: [2], [5] -> create bins with [2] and [5]

    bins = perform(MedianVarianceBinning(1), predictions, targets)
    @test length(bins) == 5
    @test all(bin -> bin.nsamples == 1, bins)
    for (i, idx) in enumerate((3, 1, 4, 2, 5))
        @test bins[i].sum_predictions == predictions[idx]
        @test bins[i].counts_targets == Matrix{Float64}(I, 3, 3)[:, targets[idx]]
    end

    bins = perform(MedianVarianceBinning(2), predictions, targets)
    @test length(bins) == 2
    for (i, idxs) in enumerate(([2, 3, 5], [1, 4]))
        @test bins[i].nsamples == length(idxs)
        @test bins[i].sum_predictions ≈ sum(predictions[idxs])
        @test bins[i].counts_targets == vec(sum(Matrix{Float64}(I, 3, 3)[:, targets[idxs]]; dims = 2))
    end

    bins = perform(MedianVarianceBinning(3), predictions, targets)
    @test length(bins) == 1
    @test bins[1].nsamples == 5
    @test bins[1].sum_predictions == sum(predictions)
    @test bins[1].counts_targets == [2, 2, 1]
end
