abstract type AbstractBinningAlgorithm end

mutable struct Bin{T}
    """Number of samples."""
    nsamples::Int
    """Mean of predictions."""
    mean_predictions::T
    """Proportions of targets."""
    proportions_targets::T

    function Bin{T}(
        nsamples::Int, mean_predictions::T, proportions_targets::T
    ) where {T<:Real}
        nsamples ≥ 0 || throw(ArgumentError("the number of samples must be non-negative"))
        return new{T}(nsamples, mean_predictions, proportions_targets)
    end

    function Bin{T}(
        nsamples::Int, mean_predictions::T, proportions_targets::T
    ) where {T<:AbstractVector{<:Real}}
        nsamples ≥ 0 || throw(ArgumentError("the number of samples must be non-negative"))
        nclasses = length(mean_predictions)
        nclasses > 1 || throw(ArgumentError("the number of classes must be greater than 1"))
        nclasses == length(proportions_targets) || throw(
            DimensionMismatch(
                "the number of predicted classes has to be equal to the number of classes",
            ),
        )

        return new{T}(nsamples, mean_predictions, proportions_targets)
    end
end

function Bin(nsamples::Int, mean_predictions::T, proportions_targets::T) where {T}
    return Bin{T}(nsamples, mean_predictions, proportions_targets)
end

"""
    Bin(predictions, targets)

Create bin of `predictions` and corresponding `targets`.
"""
function Bin(predictions::AbstractVector{<:Real}, targets::AbstractVector{Bool})
    # compute mean of predictions
    mean_predictions = mean(predictions)

    # compute proportion of targets
    proportions_targets = mean(targets)

    return Bin(length(predictions), mean_predictions, proportions_targets)
end
function Bin(
    predictions::AbstractVector{<:AbstractVector{<:Real}},
    targets::AbstractVector{<:Integer},
)
    # compute mean of predictions
    mean_predictions = mean(predictions)

    # compute proportion of targets
    nclasses = length(predictions[1])
    proportions_targets = StatsBase.proportions(targets, nclasses)

    return Bin(length(predictions), mean_predictions, proportions_targets)
end

"""
    Bin(prediction, target)

Create bin of a single `prediction` and corresponding `target`.
"""
function Bin(prediction::Real, target::Bool)
    # compute mean of predictions
    mean_predictions = prediction / 1

    # compute proportion of targets
    proportions_targets = target / 1

    return Bin(1, mean_predictions, proportions_targets)
end

function Bin(prediction::AbstractVector{<:Real}, target::Integer)
    # compute mean of predictions
    mean_predictions = prediction ./ 1

    # compute proportion of targets
    proportions_targets = similar(mean_predictions)
    for i in 1:length(proportions_targets)
        proportions_targets[i] = i == target ? 1 : 0
    end

    return Bin(1, mean_predictions, proportions_targets)
end

"""
    adddata!(bin::Bin, prediction, target)

Update running statistics of the `bin` by integrating one additional pair of `prediction`s
and `target`.
"""
function adddata!(bin::Bin, prediction::Real, target::Bool)
    @unpack mean_predictions, proportions_targets = bin

    # update number of samples
    nsamples = (bin.nsamples += 1)

    # update mean of predictions
    mean_predictions += (prediction - mean_predictions) / nsamples
    bin.mean_predictions = mean_predictions

    # update proportions of targets
    proportions_targets += (target - proportions_targets) / nsamples
    bin.proportions_targets = proportions_targets

    return nothing
end
function adddata!(bin::Bin, prediction::AbstractVector{<:Real}, target::Integer)
    @unpack mean_predictions, proportions_targets = bin

    # update number of samples
    nsamples = (bin.nsamples += 1)

    # update mean of predictions
    @. mean_predictions += (prediction - mean_predictions) / nsamples

    # update proportions of targets
    nclasses = length(proportions_targets)
    @. proportions_targets += ((1:nclasses == target) - proportions_targets) / nsamples

    return nothing
end
