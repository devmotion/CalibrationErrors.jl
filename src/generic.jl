abstract type CalibrationErrorEstimator end

function check_size(predictions::AbstractMatrix{<:Real}, labels::AbstractVector{<:Integer})
    nclasses, nsamples = size(predictions)

    nsamples == length(labels) ||
        throw(DimensionMismatch("number of predictions and labels must be equal"))
    nclasses > 1 ||
        throw(ArgumentError("the number of classes has to be greater than 1"))

    nclasses, nsamples
end

function get_predictions_labels((predictions,labels)::Tuple{<:AbstractMatrix{<:Real},
                                                            <:AbstractVector{<:Integer}})
    # check size
    check_size(predictions, labels)

    predictions, labels
end

"""
    calibrationerror(estimator, data)

Estimate the calibration error of a model from the `data` set of predicted probabilities
and corresponding labels using the `estimator`.
"""
function calibrationerror(estimator::CalibrationErrorEstimator,
                          data::Tuple{<:AbstractMatrix{<:Real},<:AbstractVector{<:Integer}})
    # check whether the number of predictions and labels is equal
    predictions, labels = get_predictions_labels(data)

    # estimate the calibration error
    _calibrationerror(estimator, predictions, labels)
end
