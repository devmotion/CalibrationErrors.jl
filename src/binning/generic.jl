abstract type AbstractBinningAlgorithm end

mutable struct Bin{T<:Real}
    """Number of samples."""
    nsamples::Int
    """Mean of predictions."""
    mean_predictions::Vector{T}
    """Proportions of targets."""
    proportions_targets::Vector{T}

    function Bin{T}(
        nsamples::Int, mean_predictions::Vector{T}, proportions_targets::Vector{T}
    ) where {T}
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

function Bin(
    nsamples::Int, mean_predictions::Vector{T}, proportions_targets::Vector{T}
) where {T<:Real}
    return Bin{T}(nsamples, mean_predictions, proportions_targets)
end

"""
    Bin(predictions, targets)

Create bin of `predictions` and corresponding `targets`.
"""
function Bin(
    predictions::AbstractVector{<:AbstractVector{<:Real}},
    targets::AbstractVector{<:Integer},
)
    # compute mean of predictions
    mean_predictions = mean(predictions)

    # compute proportion of targets
    nclasses = length(predictions[1])
    proportions_targets = proportions(targets, nclasses)

    return Bin(length(predictions), mean_predictions, proportions_targets)
end

"""
    Bin(prediction, target)

Create bin of a signle `prediction` and corresponding `target`.
"""
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
