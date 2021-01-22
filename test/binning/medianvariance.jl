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
    dist = Dirichlet(nclasses, 1.0)
    predictions = [rand(dist) for _ in 1:nsamples]
    targets = rand(1:nclasses, nsamples)

    # set minimum number of samples
    for minsize in (1, 10, 100, 500, 1_000)
        bins = @inferred(perform(MedianVarianceBinning(minsize), predictions, targets))

        @test all(bin -> bin.nsamples ≥ minsize, bins)
        @test all(bin -> sum(bin.mean_predictions) ≈ 1, bins)
        @test all(bin -> sum(bin.proportions_targets) ≈ 1, bins)

        @test sum(bin -> bin.nsamples, bins) == nsamples
        @test sum(bin -> bin.nsamples .* bin.mean_predictions, bins) ≈ sum(predictions)
        @test sum(bin -> bin.nsamples .* bin.proportions_targets, bins) ≈
              counts(targets, nclasses)
    end

    # set maximum number of bins
    for maxbins in (1, 10, 100, 500, 1_000)
        bins = @inferred(perform(MedianVarianceBinning(1, maxbins), predictions, targets))

        @test length(bins) ≤ maxbins

        @test all(bin -> bin.nsamples ≥ 1, bins)
        @test all(bin -> sum(bin.mean_predictions) ≈ 1, bins)
        @test all(bin -> sum(bin.proportions_targets) ≈ 1, bins)

        @test sum(bin -> bin.nsamples, bins) == nsamples
        @test sum(bin -> bin.nsamples .* bin.mean_predictions, bins) ≈ sum(predictions)
        @test sum(bin -> bin.nsamples .* bin.proportions_targets, bins) ≈
              counts(targets, nclasses)
    end
end

@testset "Simple example" begin
    predictions = [[0.4, 0.1, 0.5], [0.5, 0.3, 0.2], [0.3, 0.7, 0.0]]
    targets = [1, 2, 3]
    # maximum possible steps in the order they might occur:
    # first step: [1], [2, 3] -> create bin with [1]
    # second step: [2], [3] -> create bins with [2] and [3]

    bins = perform(MedianVarianceBinning(1), predictions, targets)
    @test length(bins) == 3
    @test all(bin -> bin.nsamples == 1, bins)
    for (i, idx) in enumerate((1, 2, 3))
        @test bins[i].mean_predictions == predictions[idx]
        @test bins[i].proportions_targets == Matrix{Float64}(I, 3, 3)[:, targets[idx]]
    end

    bins = perform(MedianVarianceBinning(2), predictions, targets)
    @test length(bins) == 1
    @test bins[1].nsamples == 3
    @test bins[1].mean_predictions == mean(predictions)
    @test bins[1].proportions_targets == [1 / 3, 1 / 3, 1 / 3]

    predictions = [
        [0.4, 0.1, 0.5], [0.5, 0.3, 0.2], [0.3, 0.7, 0.0], [0.1, 0.0, 0.9], [0.8, 0.1, 0.1]
    ]
    targets = [1, 2, 3, 1, 2]
    # maximum possible steps in the order they might occur:
    # first step: [3, 5], [1, 2, 4]
    # second step: [5], [3], [1, 2, 4] -> create bins with [5] and [3]
    # third step:  [2], [1, 4] -> create bin with [2]
    # fourth step: [1], [4] -> create bins with [1] and [4]

    bins = perform(MedianVarianceBinning(1), predictions, targets)
    @test length(bins) == 5
    @test all(bin -> bin.nsamples == 1, bins)
    for (i, idx) in enumerate((5, 3, 2, 1, 4))
        @test bins[i].mean_predictions == predictions[idx]
        @test bins[i].proportions_targets == Matrix{Float64}(I, 3, 3)[:, targets[idx]]
    end

    bins = perform(MedianVarianceBinning(2), predictions, targets)
    @test length(bins) == 2
    @test all(bin -> sum(bin.mean_predictions) ≈ 1, bins)
    @test all(bin -> sum(bin.proportions_targets) ≈ 1, bins)
    for (i, idxs) in enumerate(([3, 5], [1, 2, 4]))
        @test bins[i].nsamples == length(idxs)
        @test bins[i].mean_predictions ≈ mean(predictions[idxs])
        @test bins[i].proportions_targets ==
              vec(mean(Matrix{Float64}(I, 3, 3)[:, targets[idxs]]; dims=2))
    end

    bins = perform(MedianVarianceBinning(3), predictions, targets)
    @test length(bins) == 1
    @test bins[1].nsamples == 5
    @test bins[1].mean_predictions == mean(predictions)
    @test bins[1].proportions_targets == [0.4, 0.4, 0.2]
end
