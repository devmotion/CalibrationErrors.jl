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
    length(bins) > 0 || error("there must exist at least one bin")

    # evaluate distances in each bin and normalize the result
    sum(bin -> scaled_evaluate(distance, bin), bins) / nsamples
end
