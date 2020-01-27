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

    # create empty dictionary of bins
    bins = Dict{NTuple{N,Int},Bin{T}}()

    # bin all samples
    @inbounds for (prediction, target) in zip(predictions, targets)
        bin = get!(bins, binindex(prediction, nbins, nclasses), Bin{T}(N))
        adddata!(bin, prediction, target)
    end

    collect(Bin{T}, values(bins))
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