@testset "generic.jl" begin
    @testset "Simple example ($nclasses classes)" for nclasses in (2, 10, 100)
        # sample predictions and targets
        nsamples = 1_000
        dist = Dirichlet(nclasses, 1.0)
        predictions = [rand(dist) for _ in 1:nsamples]
        targets = rand(1:nclasses, nsamples)

        # create bin with all predictions and targets
        bin = CalibrationErrors.Bin(predictions, targets)

        # check statistics
        @test bin.nsamples == nsamples
        @test bin.mean_predictions ≈ mean(predictions)
        @test bin.proportions_targets == proportions(targets, nclasses)
        @test sum(bin.mean_predictions) ≈ 1
        @test sum(bin.proportions_targets) ≈ 1

        # compare with adding data
        bin2 = CalibrationErrors.Bin(predictions[1], targets[1])
        for i in 2:nsamples
            CalibrationErrors.adddata!(bin2, predictions[i], targets[i])
        end
        @test bin2.nsamples == bin.nsamples
        @test bin2.mean_predictions ≈ bin.mean_predictions
        @test bin2.proportions_targets ≈ bin.proportions_targets
    end
end
