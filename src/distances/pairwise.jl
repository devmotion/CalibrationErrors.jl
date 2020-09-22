function Distances.pairwise!(
    r::AbstractMatrix,
    d::DistributionsDistance,
    x::AbstractVector{<:Distribution},
    y::AbstractVector{<:Distribution}
)
    m = length(x)
    n = length(y)
    size(r) == (m, n) || error("output array is of incorrect size")

    @inbounds for j in 1:n
        yj = y[j]
        for i in 1:m
            r[i, j] = d(x[i], yj)
        end
    end

    return r
end

function Distances.pairwise(
    d::DistributionsDistance,
    x::AbstractVector{<:Distribution},
    y::AbstractVector{<:Distribution}
)
    r = Matrix{Distances.result_type(d, x, y)}(undef, length(x), length(y))
    return Distances.pairwise!(r, d, x, y)
end
