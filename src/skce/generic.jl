abstract type SKCE <: CalibrationErrorEstimator end

# kernel used by the estimator
kernel(skce::SKCE) = skce.kernel

# result type
function skce_result_type(skce::SKCE, predictions::AbstractMatrix{<:Real})
    s = zero(eltype(predictions))
    eltype(s^2 * zero(kernel_result_type(kernel(skce), s, s)))
end

"""
    skce_kernel(k, p, y, p̃, ỹ)

Evaluate kernel function
```math
(e_y - p)^T k(p, p̃) (e_ỹ - p̃)
```
of the squared kernel calibration error for the matrix-valued kernel `k` and predictions
`p` and `p̃` with corresponding labels `y` and `ỹ`.

This method assumes that `p`, `p̃`, `y`, and `ỹ` are valid and specified correctly, and
does not perform any checks.
"""
function skce_kernel(kernel::UniformScalingKernel, p::AbstractVector{<:Real},
                     y::Integer, p̃::AbstractVector{<:Real}, ỹ::Integer)
    @unpack λ = kernel

    # compute scalar products
    @inbounds s = dot(p, p̃) - p[ỹ] - p̃[y]
    y == ỹ && (s += 1)

    # multiply with scalar-valued kernel
    λ * s * kernel.kernel(p, p̃)
end

# evaluation of kernel function for diagonal kernels
function skce_kernel(kernel::DiagonalKernel, p::AbstractVector{<:Real},
                     y::Integer, p̃::AbstractVector{<:Real}, ỹ::Integer)
    @unpack diag = kernel

    # compute scalar products
    @inbounds s = dot(p, diag .* p̃) - p[ỹ] * diag[ỹ] - diag[y] * p̃[y]
    y == ỹ && (@inbounds s += diag[y])

    # multiply with scalar-valued kernel
    s * kernel.kernel(p, p̃)
end
