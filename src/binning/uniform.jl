struct UniformBinning <: AbstractBinningAlgorithm
    nbins::Int

    function UniformBinning(nbins::Int)
        nbins > 0 || error("number of bins must be positive")

        new(nbins)
    end
end

function perform(binning::UniformBinning,
                 predictions::AbstractVector{<:AbstractVector{T}},
                 targets::AbstractVector{<:Integer}) where {T<:Real}
    _perform(binning, predictions, targets, Val(length(predictions[1])))
end

function _perform(binning::UniformBinning,
                  predictions::AbstractVector{<:AbstractVector{T}},
                  targets::AbstractVector{<:Integer}, nclasses::Val{N}) where {T<:Real,N}
    @unpack nbins = binning

    # create bin for the initial sample
    binindices = NTuple{N,Int}[binindex(predictions[1], nbins, nclasses)]
    bins = [Bin(predictions[1], targets[1])]

    # reserve some memory (very rough guess)
    nsamples = length(predictions)
    sizehint!(bins, nsamples)
    sizehint!(binindices, nsamples)

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

    bins
end

function binindex(probs::AbstractVector{<:Real}, nbins::Int, ::Val{N})::NTuple{N,Int} where N
    ntuple(N) do i
        binindex(probs[i], nbins)
    end
end

function binindex(p::Real, nbins::Int)
    # check argument
    zero(p) ≤ p ≤ one(p) ||
        throw(ArgumentError("predictions must be between 0 and 1"))

    # handle special case p = 0
    iszero(p) && return 1

    ceil(Int, nbins * p)
end