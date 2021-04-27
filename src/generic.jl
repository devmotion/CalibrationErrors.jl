abstract type CalibrationErrorEstimator end

"""
    (estimator::CalibrationErrorEstimator)(predictions, targets)

Estimate the calibration error of a model from the set of `predictions`
and corresponding `targets` using the `estimator`.
"""
(::CalibrationErrorEstimator)(predictions, targets)

function check_nsamples(predictions, targets, min::Int=1)
    n = length(predictions)
    length(targets) == n ||
        throw(DimensionMismatch("number of predictions and targets must be equal"))
    n ≥ min || error("there must be at least ", min, min == 1 ? " sample" : " samples")
    return n
end
