struct ECE{B<:AbstractBinningAlgorithm,D} <: CalibrationErrorEstimator
    """Binning algorithm."""
    binning::B
    """Distance function."""
    distance::D
end

"""
    ECE(binning[, distance = TotalVariation()])

Estimator of the expected calibration error (ECE) for a classification model with
respect to the given `distance` function using the `binning` algorithm.

For classification models, the predictions ``P_{X_i}`` and targets ``Y_i`` are identified
with vectors in the probability simplex. The estimator of the ECE is defined as
```math
\\frac{1}{B} \\sum_{i=1}^B d\\big(\\overline{P}_i, \\overline{Y}_i\\big),
```
where ``B`` is the number of non-empty bins, ``d`` is the distance function, and
``\\overline{P}_i`` and ``\\overline{Y}_i`` are the average vector of the predictions and
the average vector of targets in the ``i``th bin. By default, the total variation distance
is used.

The `distance` has to be a function of the form
```julia
distance(pbar::Vector{<:Real}, ybar::Vector{<:Real}).
```
In particular, distance measures of the package
[Distances.jl](https://github.com/JuliaStats/Distances.jl) are supported.
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
        x = distance(bin.mean_predictions, bin.proportions_targets)

        # initialize the estimate
        estimate = x / 1

        # for all other bins
        n = bin.nsamples
        for i in 2:nbins
            # evaluate the distance
            bin = bins[i]
            x = distance(bin.mean_predictions, bin.proportions_targets)

            # update the estimate
            m = bin.nsamples
            n += m
            estimate += (m / n) * (x - estimate)
        end
    end

    estimate
end
