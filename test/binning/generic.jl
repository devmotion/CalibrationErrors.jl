using CalibrationErrors, Distances, Distributions, StatsBase
using CalibrationErrors: Bin, scaled_evaluate, adddata!

using Random
using Test

Random.seed!(1234)

@testset "Simple example ($nclasses classes)" for nclasses in (2, 10, 100)
    # sample predictions and targets
    nsamples = 1_000
    dist = Dirichlet(nclasses, 1)
    predictions = [rand(dist) for _ in 1:nsamples]
    targets = rand(1:nclasses, nsamples)

    # create bin with all predictions and targets
    bin = Bin(predictions, targets)

    # check statistics
    @test bin.nsamples == nsamples
    @test bin.sum_predictions ≈ sum(predictions)
    @test bin.counts_targets == counts(targets, nclasses)

    # check distance calculations
    for distance in (TotalVariation(), Cityblock(), Euclidean(), SqEuclidean())
        @test scaled_evaluate(distance, bin) ≈
            nsamples * evaluate(distance, mean(predictions),
                                proportions(targets, nclasses))
    end

    # compare with adding data to empty bin
    bin2 = Bin{eltype(predictions[1])}(nclasses)
    for (prediction, target) in zip(predictions, targets)
        adddata!(bin2, prediction, target)
    end
    @test bin2.nsamples == bin.nsamples
    @test bin2.sum_predictions == bin.sum_predictions
    @test bin2.counts_targets == bin.counts_targets
end
