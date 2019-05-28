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
function _calibrationerror(ece::ECE, predictions::AbstractMatrix{<:Real},
                           labels::AbstractVector{<:Integer})
    @unpack binning, distance = ece

    # bin predictions and labels
    bins = perform(binning, predictions, labels)

    # compute output type of estimates in each bin
    T = typeof(bin_eval_end(distance,
                            zero(result_type(distance, predictions, Int[])), 1))

    # compute weighted sum of distances
    d = zero(T)
    for bin in values(bins)
        d += bin_eval(bin, distance)
    end

    # normalize result
    inv(length(labels)) * d
end
