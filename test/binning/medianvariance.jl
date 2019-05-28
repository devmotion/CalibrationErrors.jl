using CalibrationErrors, Distributions, StatsBase
using CalibrationErrors: perform

using Random

Random.seed!(1234)

@testset "Simple example ($nclasses classes)" for nclasses in (2, 10, 100)
    nsamples = 1_000
    predictions = rand(Dirichlet(nclasses, 1), nsamples)
    labels = rand(1:nclasses, nsamples)

    # check number of samples
    bins = perform(MedianVarianceBinning(10), predictions, labels)
    for (idx, bin) in bins
        @test bin.nsamples[] >= 5
    end

    # check number of bins
    bins = perform(MedianVarianceBinning(0, 10), predictions, labels)
    @test length(bins) <= 10
end
