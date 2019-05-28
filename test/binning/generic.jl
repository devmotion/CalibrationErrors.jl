using CalibrationErrors, Distances, Distributions, StatsBase
using CalibrationErrors: Bin, bin_eval, adddata!

using Random

Random.seed!(1234)

@testset "Simple example ($nclasses classes)" for nclasses in (2, 10, 100)
    # sample predictions and labels
    nsamples = 1_000
    predictions = rand(Dirichlet(nclasses, 1), nsamples)
    labels = rand(1:nclasses, nsamples)

    # create bin with all predictions and labels
    bin = Bin(predictions, labels)

    # check statistics
    @test bin.nsamples[] == nsamples
    @test bin.prediction_sum ≈ vec(sum(predictions, dims=2))
    @test bin.label_counts == counts(labels, nclasses)

    # check distance calculations
    for distance in (TotalVariation(), Cityblock(), Euclidean(), SqEuclidean())
        @test bin_eval(bin, distance) ≈
            nsamples * evaluate(distance, vec(mean(predictions; dims=2)),
                                proportions(labels, nclasses))
    end

    # compare with adding data to empty bin
    bin2 = Bin(eltype(predictions), nclasses)
    adddata!(bin2, predictions, labels)
    @test bin2.nsamples[] == bin.nsamples[]
    @test bin2.prediction_sum == bin.prediction_sum
    @test bin2.label_counts == bin.label_counts

    bin3 = Bin(eltype(predictions), nclasses)
    for i in axes(predictions, 2)
        adddata!(bin3, view(predictions, :, i), labels[i])
    end
    @test bin3.nsamples[] == bin.nsamples[]
    @test bin3.prediction_sum == bin.prediction_sum
    @test bin3.label_counts == bin.label_counts
end
