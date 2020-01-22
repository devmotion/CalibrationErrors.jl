abstract type CalibrationErrorEstimator end

# extract predictions and targets from tuple
predictions_targets((predictions, targets)::Tuple{<:Any,<:Any}) =
    predictions_targets(predictions, targets)

# extract predictions and targets from a vector of tuples
predictions_targets(data::AbstractVector{<:Tuple{<:Any,<:Any}}) =
    predictions_targets(first.(data), last.(data))

# extract predictions from a matrix
predictions_targets(predictions::AbstractMatrix{<:Real}, targets::AbstractVector) =
    predictions_targets([predictions[:, i] for i in axes(predictions, 2)], targets)

# do not transform vectors
function predictions_targets(predictions::AbstractVector, targets::AbstractVector)
    predictions, targets
end

"""
    calibrationerror(estimator::CalibrationErrorEstimator, data...)

Estimate the calibration error of a model from the `data` set of predictions
and corresponding targets using the `estimator`.

The `data` can be a tuple of predictions and targets or an array of tuples of
predictions and targets.
"""
function calibrationerror(estimator::CalibrationErrorEstimator, data...)
    predictions, targets = predictions_targets(data...)

    length(predictions) == length(targets) ||
        throw(DimensionMismatch("number of predictions and targets must be equal"))

    _calibrationerror(estimator, predictions, targets)
end