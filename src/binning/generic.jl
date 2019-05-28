abstract type AbstractBinningAlgorithm end

struct Bin{T<:Real}
    """Number of samples."""
    nsamples::Ref{Int}
    """Sum of predictions."""
    prediction_sum::Vector{T}
    """Counts of labels."""
    label_counts::Vector{Int}

    function Bin{T}(nsamples::Ref{Int}, prediction_sum::Vector{T},
                    label_counts::Vector{Int}) where T
        nsamples[] ≥ 0 || throw(ArgumentError("the number of samples must be non-negative"))
        nclasses = length(prediction_sum)
        nclasses > 1 || throw(ArgumentError("the number of classes must be greater than 1"))
        nclasses == length(label_counts) ||
            throw(DimensionMismatch("the number of predicted classes has to be equal to the number of classes"))

        new{T}(nsamples, prediction_sum, label_counts)
    end
end

Bin(nsamples::Ref{Int}, prediction_sum::Vector{<:Real}, label_counts::Vector{Int}) =
    Bin{eltype(prediction_sum)}(nsamples, prediction_sum, label_counts)

"""
    Bin(predictions, labels)

Create bin of `predictions` and corresponding `labels`.
"""
function Bin(predictions::AbstractMatrix{<:Real}, labels::AbstractVector{<:Integer})
    # check arguments
    nclasses, nsamples = check_size(predictions, labels)

    # compute sum of predictions
    prediction_sum = vec(sum(predictions, dims=2))

    # compute counts of labels
    label_counts = counts(labels, 1:nclasses)

    Bin(Ref(nsamples), prediction_sum, label_counts)
end

"""
    Bin(T, m::Int)

Create empty container that keeps track of the sum of predictions of type `T` and the
counts of labels of a model with `m` classes.
"""
Bin(::Type{T}, m::Int) where T = Bin(Ref(0), zeros(T, m), zeros(Int, m))

"""
    adddata!(bin::Bin, predictions, labels)

Update running statistics of the `bin` by integrating additional pair(s) of `prediction`s
and `label`s.
"""
function adddata!(bin::Bin, prediction::AbstractVector{<:Real}, label::Integer)
    @unpack nsamples, prediction_sum, label_counts = bin

    # check arguments
    length(prediction) == length(prediction_sum) ||
        throw(DimensionMismatch("the number of predicted classes is incorrect"))

    # update number of samples
    nsamples[] += 1

    # update sum of predictions
    prediction_sum .+= prediction

    # update counts of labels
    if 1 ≤ label ≤ length(label_counts)
        @inbounds label_counts[label] += 1
    end

    nothing
end

function adddata!(bin::Bin, predictions::AbstractMatrix{<:Real},
                  labels::AbstractVector{<:Integer})
    @unpack nsamples, prediction_sum, label_counts = bin

    # check arguments
    addnclasses, addnsamples = check_size(predictions, labels)
    addnclasses == length(prediction_sum) ||
        throw(DimensionMismatch("the number of predicted classes is incorrect"))

    # update number of samples
    nsamples[] += addnsamples

    # update sum of predictions
    for p in eachcol(predictions)
        prediction_sum .+= p
    end

    # update counts of labels
    addcounts!(label_counts, labels, 1:addnclasses)

    nothing
end

# distance computations

"""
    bin_eval(bin::Bin, distance)

Evaluate `distance` between the average prediction and the distribution of labels in the
`bin`, multiplied by the number of samples.

The multiplication with the number of samples simplifies the computation of the expected
miscalibration error (ECE)[@ref].
"""
function bin_eval(bin::Bin, distance::Union{TotalVariation,Cityblock,Euclidean,SqEuclidean})
    @unpack nsamples, prediction_sum, label_counts = bin

    # compute output type
    T = typeof(bin_eval_end(distance,
                            zero(result_type(distance, prediction_sum, label_counts)), 1))

    # if bin is empty return 0
    iszero(nsamples[]) && return zero(T)

    # compute distance between sum of predictions and counts of labels
    d = evaluate(distance, prediction_sum, label_counts)

    # perform normalization (e.g., in the case of squared Euclidean distance)
    bin_eval_end(distance, d, nsamples[])::T
end

# normalization is required since the distance is computed between sum of predictions
# and counts of labels instead of between the mean of predictions and distribution of labels
# thus in the case of the squared Euclidean distance without normalization, the result
# would be multiplied by the squared number of samples instead of only the number of
# samples.
bin_eval_end(::Union{TotalVariation,Cityblock,Euclidean}, d::Real, ::Int) = d
bin_eval_end(::SqEuclidean, d::Real, nsamples::Int) = d * inv(nsamples)
