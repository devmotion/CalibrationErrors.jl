struct UnnormalizedMKCE{K<:MatrixKernel,V<:AbstractArray} <: CalibrationErrorEstimator
    kernel::K
    witness::V
end

function _calibrationerror(estimator::UnnormalizedMKCE, predictions::AbstractMatrix{<:Real},
                          labels::AbstractVector{<:Integer})
    @unpack kernel, witness = estimator

    # obtain number of samples
    nclasses, nsamples = size(predictions)
    nwitness = length(witness)

    T = float(eltype(predictions))
    cache = Vector{T}(undef, nclasses)
    meancache = Vector{T}(undef, nclasses)

    s = zero(T)
    for i in 1:nwitness
        v = witness[i]
        fill!(meancache, zero(T))

        for j in 1:nsamples
            mkce_kernel!(cache, kernel, v, view(predictions, :, j), labels[j])
            meancache .+= cache
        end
        meancache ./= nsamples

        s += sum(abs2, meancache)
    end

    s / nwitness
end

"""
    mkce_kernel!(out, k, v, p, y)

Evaluate kernel function
```math
k(v, p) (p - e_y)
```
of the mean kernel calibration error for the matrix-valued kernel `k`, witness point `v`
and prediction `p` with corresponding label `y`.

This method assumes that `v`, `p`, and `y` are valid and specified correctly, and
does not perform any checks.
"""
function mkce_kernel!(out, kernel::MatrixKernel, v::AbstractVector{<:Real},
                      p::AbstractVector{<:Real}, y::Integer)
    copyto!(out, p)
    @inbounds out[y] -= 1
    lmul!(kernel(v, p), out)

    out
end

function mkce_kernel!(out, kernel::UniformScalingKernel, v::AbstractVector{<:Real},
                      p::AbstractVector{<:Real}, y::Integer)
    @unpack λ = kernel

    copyto!(out, p)
    @inbounds out[y] -= 1
    lmul!(λ * kernel.kernel(v, p), out)
    
    out
end

function mkce_kernel!(out, kernel::DiagonalKernel, v::AbstractVector{<:Real},
                      p::AbstractVector{<:Real}, y::Integer)
    @unpack diag = kernel

    copyto!(out, p)
    @inbounds out[y] -= 1
    k = kernel.kernel(v, p)
    @. out = k * diag
    
    out
end