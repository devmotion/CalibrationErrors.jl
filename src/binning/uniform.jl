"""
    UniformBinning(nbins::Int)

Binning scheme of the probability simplex with `nbins` bins of uniform width for
each component.
"""
struct UniformBinning <: AbstractBinningAlgorithm
    nbins::Int

    function UniformBinning(nbins::Int)
        nbins > 0 || error("number of bins must be positive")

        return new(nbins)
    end
end

function perform(
    binning::UniformBinning,
    predictions::AbstractVector{<:Real},
    targets::AbstractVector{Bool},
)
    @unpack nbins = binning

    # create dictionary of bins
    T = eltype(float(zero(eltype(predictions))))
    bins = Dict{Int,Bin{T}}()

    # reserve some memory (very rough guess)
    nsamples = length(predictions)
    sizehint!(bins, min(nbins, nsamples))

    # for all other samples
    @inbounds for (prediction, target) in zip(predictions, targets)
        # compute index of bin
        index = binindex(prediction, nbins)

        # create new bin or update existing one
        bin = get(bins, index, nothing)
        if bin === nothing
            bins[index] = Bin(prediction, target)
        else
            adddata!(bin, prediction, target)
        end
    end

    return values(bins)
end

function perform(
    binning::UniformBinning,
    predictions::AbstractVector{<:AbstractVector{<:Real}},
    targets::AbstractVector{<:Integer},
)
    return _perform(binning, predictions, targets, Val(length(predictions[1])))
end

function _perform(
    binning::UniformBinning,
    predictions::AbstractVector{<:AbstractVector{T}},
    targets::AbstractVector{<:Integer},
    nclasses::Val{N},
) where {T<:Real,N}
    @unpack nbins = binning

    # create bin for the initial sample
    binindices = NTuple{N,Int}[binindex(predictions[1], nbins, nclasses)]
    bins = [Bin(predictions[1], targets[1])]

    # reserve some memory (very rough guess)
    nsamples = length(predictions)
    guess = min(nbins, nsamples)
    sizehint!(bins, guess)
    sizehint!(binindices, guess)

    # for all other samples
    @inbounds for i in 2:nsamples
        # obtain prediction and corresponding target
        prediction = predictions[i]
        target = targets[i]

        # compute index of bin
        index = binindex(prediction, nbins, nclasses)

        # create new bin or update existing one
        j = searchsortedfirst(binindices, index)
        if j > length(binindices) || (binindices[j] !== index)
            insert!(binindices, j, index)
            insert!(bins, j, Bin(prediction, target))
        else
            bin = bins[j]
            adddata!(bin, prediction, target)
        end
    end

    return bins
end

function binindex(
    probs::AbstractVector{<:Real}, nbins::Int, ::Val{N}
)::NTuple{N,Int} where {N}
    ntuple(N) do i
        binindex(probs[i], nbins)
    end
end

function binindex(p::Real, nbins::Int)
    # check argument
    zero(p) ≤ p ≤ one(p) || throw(ArgumentError("predictions must be between 0 and 1"))

    # handle special case p = 0
    iszero(p) && return 1

    return ceil(Int, nbins * p)
end
