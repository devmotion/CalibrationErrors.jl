using CalibrationErrors, Distributions, StatsBase
using CalibrationErrors: perform, binindex

using Random
using Test

Random.seed!(1234)

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

@testset "Simple example ($nclasses classes)" for nclasses in (2, 10, 100)
    # sample predictions and targets
    nsamples = 1_000
    dist = Dirichlet(nclasses, 1)
    predictions = [rand(dist) for _ in 1:nsamples]
    targets = rand(1:nclasses, nsamples)

    # bin data in bins of uniform width
    bins = @inferred(perform(UniformBinning(10), predictions, targets))

    # check all bins
    for bin in bins
        # compute index of bin from average prediction
        idx = binindex(bin.sum_predictions ./ bin.nsamples, 10, Val(nclasses))

        # compute indices of all predictions in the same bin
        idxs = filter(i -> idx == binindex(predictions[i], 10, Val(nclasses)), 1:nsamples)

        @test bin.nsamples == length(idxs)
        @test bin.sum_predictions ≈ sum(predictions[idxs])
        @test bin.counts_targets ≈ counts(targets[idxs], 1:nclasses)
    end
end
