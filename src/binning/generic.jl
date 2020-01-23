abstract type AbstractBinningAlgorithm end

mutable struct Bin{T<:Real}
    """Number of samples."""
    nsamples::Int
    """Sum of predictions."""
    sum_predictions::Vector{T}
    """Counts of targets."""
    counts_targets::Vector{Int}

    function Bin{T}(nsamples::Int, sum_predictions::Vector{T},
                    counts_targets::Vector{Int}) where {T}
        nsamples ≥ 0 || throw(ArgumentError("the number of samples must be non-negative"))
        nclasses = length(sum_predictions)
        nclasses > 1 || throw(ArgumentError("the number of classes must be greater than 1"))
        nclasses == length(counts_targets) ||
            throw(DimensionMismatch("the number of predicted classes has to be equal to the number of classes"))

        new{T}(nsamples, sum_predictions, counts_targets)
    end
end

Bin(nsamples::Int, sum_predictions::Vector{<:Real}, counts_targets::Vector{Int}) =
    Bin{eltype(sum_predictions)}(nsamples, sum_predictions, counts_targets)

function Bin(nsamples::Int, sum_predictions::AbstractVector{<:Real},
             counts_targets::Vector{Int})
    Bin(nsamples, convert(Vector, sum_predictions), counts_targets)
end

"""
    Bin(predictions, targets)

Create bin of `predictions` and corresponding `targets`.
"""
function Bin(predictions::AbstractVector{<:AbstractVector{<:Real}},
             targets::AbstractVector{<:Integer})
    # compute sum of predictions
    sum_predictions = sum(predictions)

    # compute counts of targets
    nclasses = length(predictions[1])
    counts_targets = counts(targets, nclasses)

    Bin(length(predictions), sum_predictions, counts_targets)
end

"""
    Bin{T}(nclasses::Int)

Create empty container that keeps track of the sum of predictions of type `T` and the
counts of targets of a model with `nclasses` classes.
"""
Bin{T}(nclasses::Int) where {T<:Real} = Bin(0, zeros(T, nclasses), zeros(Int, nclasses))

"""
    adddata!(bin::Bin, prediction, target)

Update running statistics of the `bin` by integrating one additional pair of `prediction`s
and `target`.
"""
function adddata!(bin::Bin, prediction::AbstractVector{<:Real}, target::Integer)
    @unpack sum_predictions, counts_targets = bin

    # update number of samples
    bin.nsamples += 1

    # update sum of predictions
    sum_predictions .+= prediction

    # update counts of targets
    if 1 ≤ target ≤ length(counts_targets)
        @inbounds counts_targets[target] += 1
    end

    nothing
end

# distance computations

"""
    scaled_evaluate(distance, bin::Bin)

Evaluate `distance` between the average prediction and the distribution of targets in the
`bin`, multiplied by the number of samples.

The multiplication with the number of samples simplifies the computation of the expected
miscalibration error (ECE)[@ref].
"""
function scaled_evaluate(distance::Union{TotalVariation,Cityblock,Euclidean,SqEuclidean},
                         bin::Bin)
    @unpack nsamples, sum_predictions, counts_targets = bin

    # handle empty bins
    n = iszero(nsamples) ? 1 : nsamples

    # compute distance between sum of predictions and counts of targets
    d = evaluate(distance, sum_predictions, counts_targets)

    # perform normalization (e.g., in the case of squared Euclidean distance)
    normalize_distance(distance, d, n)
end

# Normalization is required since the distance is computed between sum of predictions
# and counts of targets instead of between the mean of predictions and distribution of
# targets. Thus in the case of the squared Euclidean distance without normalization, the
# result would be multiplied by the squared number of samples instead of only the number of
# samples.
normalize_distance(::Union{TotalVariation,Cityblock,Euclidean}, d::Real, ::Int) = d
normalize_distance(::SqEuclidean, d::Real, nsamples::Int) = d / nsamples
