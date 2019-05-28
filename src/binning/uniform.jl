struct UniformBinning <: AbstractBinningAlgorithm
    nbins::Int
end

function perform(binning::UniformBinning, predictions::AbstractMatrix{<:Real},
                 labels::AbstractVector{<:Integer})
    # create empty dictionary of bins
    T = eltype(predictions)
    bins = Dict{Int,Bin{T}}()

    # bin all samples
    nclasses = size(predictions, 1)
    @inbounds for i in 1:length(labels)
        prediction = view(predictions, :, i)
        bin = get!(bins, binid(binning, prediction), Bin(T, nclasses))
        adddata!(bin, prediction, labels[i])
    end

    bins
end

function binid(alg::UniformBinning, prediction::AbstractVector{<:Real})
    id = 1
    @inbounds for p in prediction
        id *= binid(alg, p)
    end
    id
end

function binid(alg::UniformBinning, p::Real)
    # check argument
    (p < zero(p) || p > one(p)) &&
        throw(ArgumentError("predictions must be between 0 and 1"))

    # handle special case p = 0
    iszero(p) && return 1

    ceil(alg.nbins * p)
end
