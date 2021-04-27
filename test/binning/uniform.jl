@testset "uniform.jl" begin
    @testset "Constructor" begin
        @test_throws ErrorException UniformBinning(-1)
        @test_throws ErrorException UniformBinning(0)
    end

    @testset "Binning indices" begin
        # scalars
        @test_throws ArgumentError CalibrationErrors.binindex(-0.5, 10)
        @test CalibrationErrors.binindex(0, 10) == 1
        @test CalibrationErrors.binindex(0.1, 10) == 1
        @test CalibrationErrors.binindex(0.45, 10) == 5
        @test CalibrationErrors.binindex(1, 10) == 10
        @test_throws ArgumentError CalibrationErrors.binindex(1.5, 10)

        # vectors
        @test_throws ArgumentError CalibrationErrors.binindex([-0.5, 0.5], 10, Val(2))
        @test @inferred(CalibrationErrors.binindex([0, 0], 10, Val(2))) == (1, 1)
        @test @inferred(CalibrationErrors.binindex([0.1, 0], 10, Val(2))) == (1, 1)
        @test @inferred(CalibrationErrors.binindex([0.45, 0.55], 10, Val(2))) == (5, 6)
        @test @inferred(CalibrationErrors.binindex([1, 1], 10, Val(2))) == (10, 10)
        @test_throws ArgumentError CalibrationErrors.binindex([1.5, 0.5], 10, Val(2))
    end

    @testset "Basic tests ($nclasses classes)" for nclasses in (2, 10, 100)
        # sample predictions and targets
        nsamples = 1_000
        dist = Dirichlet(nclasses, 1.0)
        predictions = [rand(dist) for _ in 1:nsamples]
        targets = rand(1:nclasses, nsamples)

        for nbins in (1, 10, 100, 500, 1_000)
            # bin data in bins of uniform width
            bins = @inferred(
                CalibrationErrors.perform(UniformBinning(nbins), predictions, targets)
            )

            # check all bins
            for bin in bins
                # compute index of bin from average prediction
                idx = CalibrationErrors.binindex(bin.mean_predictions, nbins, Val(nclasses))

                # compute indices of all predictions in the same bin
                idxs = filter(
                    i ->
                        idx ==
                        CalibrationErrors.binindex(predictions[i], nbins, Val(nclasses)),
                    1:nsamples,
                )

                @test bin.nsamples == length(idxs)
                @test bin.mean_predictions ≈ mean(predictions[idxs])
                @test bin.proportions_targets ≈ proportions(targets[idxs], 1:nclasses)
            end
        end
    end

    @testset "Simple example" begin
        predictions = [[0.4, 0.1, 0.5], [0.5, 0.3, 0.2], [0.3, 0.7, 0.0]]
        targets = [1, 2, 3]

        bins = CalibrationErrors.perform(UniformBinning(2), predictions, targets)
        @test length(bins) == 2
        sort!(bins; by=x -> x.nsamples)
        @test all(bin -> sum(bin.mean_predictions) == 1, bins)
        @test all(bin -> sum(bin.proportions_targets) == 1, bins)
        for (i, idxs) in enumerate(([3], [1, 2]))
            @test bins[i].nsamples == length(idxs)
            @test bins[i].mean_predictions == mean(predictions[idxs])
            @test bins[i].proportions_targets ==
                  vec(mean(Matrix{Float64}(I, 3, 3)[:, targets[idxs]]; dims=2))
        end

        bins = CalibrationErrors.perform(UniformBinning(1), predictions, targets)
        @test length(bins) == 1
        @test bins[1].nsamples == 3
        @test bins[1].mean_predictions ≈ mean(predictions)
        @test bins[1].proportions_targets ≈ [1 / 3, 1 / 3, 1 / 3]

        predictions = [
            [0.4, 0.1, 0.5],
            [0.5, 0.3, 0.2],
            [0.3, 0.7, 0.0],
            [0.1, 0.0, 0.9],
            [0.8, 0.1, 0.1],
        ]
        targets = [1, 2, 3, 1, 2]

        bins = CalibrationErrors.perform(UniformBinning(3), predictions, targets)
        sort!(bins; by=x -> x.mean_predictions[1])
        @test length(bins) == 5
        @test all(bin -> bin.nsamples == 1, bins)
        for (i, idx) in enumerate((4, 3, 1, 2, 5))
            @test bins[i].mean_predictions == predictions[idx]
            @test bins[i].proportions_targets == Matrix{Float64}(I, 3, 3)[:, targets[idx]]
        end

        bins = CalibrationErrors.perform(UniformBinning(2), predictions, targets)
        sort!(bins; by=x -> x.mean_predictions[1])
        @test length(bins) == 4
        @test all(bin -> sum(bin.mean_predictions) == 1, bins)
        @test all(bin -> sum(bin.proportions_targets) == 1, bins)
        for (i, idxs) in enumerate(([4], [3], [1, 2], [5]))
            @test bins[i].nsamples == length(idxs)
            @test bins[i].mean_predictions == mean(predictions[idxs])
            @test bins[i].proportions_targets ==
                  vec(mean(Matrix{Float64}(I, 3, 3)[:, targets[idxs]]; dims=2))
        end

        bins = CalibrationErrors.perform(UniformBinning(1), predictions, targets)
        @test length(bins) == 1
        @test bins[1].nsamples == 5
        @test bins[1].mean_predictions ≈ mean(predictions)
        @test bins[1].proportions_targets == [0.4, 0.4, 0.2]
    end
end
