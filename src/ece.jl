struct ECE{B<:AbstractBinningAlgorithm,D} <: CalibrationErrorEstimator
    """Binning algorithm."""
    binning::B
    """Distance function."""
    distance::D
end

"""
    ECE(binning[, distance = TotalVariation()])

Create an estimator of the expected calibration error (ECE) with the given `binning`
algorithm and `distance` function.
"""
ECE(binning::AbstractBinningAlgorithm) = ECE(binning, TotalVariation())

# estimate ECE
function _calibrationerror(ece::ECE, predictions::AbstractVector{<:AbstractVector{<:Real}},
                           targets::AbstractVector{<:Integer})
    @unpack binning, distance = ece

    # check number of samples
    nsamples = length(predictions)
    nsamples > 0 || error("there must be at least one sample")

    # bin predictions and labels
    bins = perform(binning, predictions, targets)
    nbins = length(bins)
    nbins > 0 || error("there must exist at least one bin")

    # compute the weighted mean of the distances in each bin
    # use West's algorithm for numerical stability

    # evaluate the distance in the first bin
    @inbounds begin
        bin = bins[1]
        x = evaluate(distance, bin)

        # initialize the estimate
        estimate = x / 1

        # for all other bins
        n = bin.nsamples
        for i in 2:nbins
            # evaluate the distance
            bin = bins[i]
            x = evaluate(distance, bin)

            # update the estimate
            m = bin.nsamples
            n += m
            estimate += (m / n) * (x - estimate)
        end
    end

    estimate
end
