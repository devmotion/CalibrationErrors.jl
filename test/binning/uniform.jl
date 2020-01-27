using CalibrationErrors, Distributions, StatsBase
using CalibrationErrors: perform, binindex

using LinearAlgebra
using Random
using Test

Random.seed!(1234)

@testset "Constructor" begin
    @test_throws ErrorException UniformBinning(-1)
    @test_throws ErrorException UniformBinning(0)
end

@testset "Binning indices" begin
    # binindex with scalars
    @test_throws ArgumentError binindex(-0.5, 10)
    @test binindex(0, 10) == 1
    @test binindex(0.1, 10) == 1
    @test binindex(0.45, 10) == 5
    @test binindex(1, 10) == 10
    @test_throws ArgumentError binindex(1.5, 10)

    # binindex with vectors
    @test_throws ArgumentError binindex([-0.5, 0.5], 10, Val(2))
    @test @inferred(binindex([0, 0], 10, Val(2))) == (1, 1)
    @test @inferred(binindex([0.1, 0], 10, Val(2))) == (1, 1)
    @test @inferred(binindex([0.45, 0.55], 10, Val(2))) == (5, 6)
    @test @inferred(binindex([1, 1], 10, Val(2))) == (10, 10)
    @test_throws ArgumentError binindex([1.5, 0.5], 10, Val(2))
end

@testset "Basic tests ($nclasses classes)" for nclasses in (2, 10, 100)
    # sample predictions and targets
    nsamples = 1_000
    dist = Dirichlet(nclasses, 1)
    predictions = [rand(dist) for _ in 1:nsamples]
    targets = rand(1:nclasses, nsamples)

    for nbins in (1, 10, 100, 500, 1_000)
        # bin data in bins of uniform width
        bins = @inferred(perform(UniformBinning(nbins), predictions, targets))

        # check all bins
        for bin in bins
            # compute index of bin from average prediction
            idx = binindex(bin.sum_predictions ./ bin.nsamples, nbins, Val(nclasses))

            # compute indices of all predictions in the same bin
            idxs = filter(i -> idx == binindex(predictions[i], nbins, Val(nclasses)), 1:nsamples)

            @test bin.nsamples == length(idxs)
            @test bin.sum_predictions ≈ sum(predictions[idxs])
            @test bin.counts_targets ≈ counts(targets[idxs], 1:nclasses)
        end
    end
end

@testset "Simple example" begin
    predictions = [[0.4, 0.1, 0.5], [0.5, 0.3, 0.2], [0.3, 0.7, 0.0]]
    targets = [1, 2, 3]

    bins = perform(UniformBinning(2), predictions, targets)
    @test length(bins) == 2
    sort!(bins; by = x -> x.nsamples)
    for (i, idxs) in enumerate(([3], [1, 2]))
        @test bins[i].nsamples == length(idxs)
        @test bins[i].sum_predictions == sum(predictions[idxs])
        @test bins[i].counts_targets == vec(sum(Matrix{Float64}(I, 3, 3)[:, targets[idxs]]; dims = 2))
    end

    bins = perform(UniformBinning(1), predictions, targets)
    @test length(bins) == 1
    @test bins[1].nsamples == 3
    @test bins[1].sum_predictions == sum(predictions)
    @test bins[1].counts_targets == [1, 1, 1]

    predictions = [[0.4, 0.1, 0.5], [0.5, 0.3, 0.2], [0.3, 0.7, 0.0], [0.1, 0.0, 0.9], [0.8, 0.1, 0.1]]
    targets = [1, 2, 3, 1, 2]

    bins = perform(UniformBinning(3), predictions, targets)
    sort!(bins; by = x -> x.sum_predictions[1])
    @test length(bins) == 5
    @test all(bin -> bin.nsamples == 1, bins)
    for (i, idx) in enumerate((4, 3, 1, 2, 5))
        @test bins[i].sum_predictions == predictions[idx]
        @test bins[i].counts_targets == Matrix{Float64}(I, 3, 3)[:, targets[idx]]
    end

    bins = perform(UniformBinning(2), predictions, targets)
    sort!(bins; by = x -> x.sum_predictions[1])
    @test length(bins) == 4
    for (i, idxs) in enumerate(([4], [3], [5], [1, 2]))
        @test bins[i].nsamples == length(idxs)
        @test bins[i].sum_predictions == sum(predictions[idxs])
        @test bins[i].counts_targets == vec(sum(Matrix{Float64}(I, 3, 3)[:, targets[idxs]]; dims = 2))
    end

    bins = perform(UniformBinning(1), predictions, targets)
    @test length(bins) == 1
    @test bins[1].nsamples == 5
    @test bins[1].sum_predictions == sum(predictions)
    @test bins[1].counts_targets == [2, 2, 1]
end
