abstract type SKCE <: CalibrationErrorEstimator end

"""
    unsafe_skce_eval(k, p, y, p̃, ỹ)

Evaluate
```math
k((p, y), (p̃, ỹ)) - E_{z ∼ p}[k((p, z), (p̃, ỹ))] - E_{z̃ ∼ p̃}[k((p, y), (p̃, z̃))] + E_{z ∼ p, z̃ ∼ p̃}[k((p, z), (p̃, z̃))]
```
for kernel `k` and predictions `p` and `p̃` with corresponding targets `y` and `ỹ`.

This method assumes that `p`, `p̃`, `y`, and `ỹ` are valid and specified correctly, and
does not perform any checks.
"""
function unsafe_skce_eval end

# default implementation for classification
# we do not use the symmetry of `kernel` since it seems unlikely that `(p, y) == (p̃, ỹ)`
function unsafe_skce_eval(
    kernel::Kernel,
    p::AbstractVector{<:Real},
    y::Integer,
    p̃::AbstractVector{<:Real},
    ỹ::Integer
)
    # precomputations
    n = length(p)

    @inbounds py = p[y]
    @inbounds p̃ỹ = p̃[ỹ]
    pym1 = py - 1
    p̃ỹm1 = p̃ỹ - 1

    tuple_p_y = (p, y)
    tuple_p̃_ỹ = (p̃, ỹ)

    # i = y, j = ỹ
    result = kernel((p, y), (p̃, ỹ)) * (1 - py - p̃ỹ + py * p̃ỹ)

    # i < y
    for i in 1:(y - 1)
        @inbounds pi = p[i]
        tuple_p_i = (p, i)

        # j < ỹ
        @inbounds for j in 1:(ỹ - 1)
            result += kernel(tuple_p_i, (p̃, j)) * pi * p̃[j]
        end

        # j = ỹ
        result += kernel(tuple_p_i, tuple_p̃_ỹ) * pi * p̃ỹm1

        # j > ỹ
        @inbounds for j in (ỹ + 1):n
            result += kernel(tuple_p_i, (p̃, j)) * pi * p̃[j]
        end
    end

    # i = y, j < ỹ
    @inbounds for j in 1:(ỹ - 1)
        result += kernel(tuple_p_y, (p̃, j)) * pym1 * p̃[j]
    end

    # i = y, j > ỹ
    @inbounds for j in (ỹ + 1):n
        result += kernel(tuple_p_y, (p̃, j)) * pym1 * p̃[j]
    end

    # i > y
    for i in (y + 1):n
        @inbounds pi = p[i]
        tuple_p_i = (p, i)

        # j < ỹ
        @inbounds for j in 1:(ỹ - 1)
            result += kernel(tuple_p_i, (p̃, j)) * pi * p̃[j]
        end

        # j = ỹ
        result += kernel(tuple_p_i, tuple_p̃_ỹ) * pi * p̃ỹm1

        # j > ỹ
        @inbounds for j in (ỹ + 1):n
            result += kernel(tuple_p_i, (p̃, j)) * pi * p̃[j]
        end
    end

    result
end

# evaluation for tensor product kernels
function unsafe_skce_eval(kernel::TensorProduct, p, y, p̃, ỹ)
    κpredictions, κtargets = kernel.kernels
    return κpredictions(p, p̃) * unsafe_skce_eval_targets(κtargets, p, y, p̃, ỹ)
end

# resolve method ambiguity
function unsafe_skce_eval(
    kernel::TensorProduct,
    p::AbstractVector{<:Real},
    y::Integer,
    p̃::AbstractVector{<:Real},
    ỹ::Integer
)
    κpredictions, κtargets = kernel.kernels
    return κpredictions(p, p̃) * unsafe_skce_eval_targets(κtargets, p, y, p̃, ỹ)
end

function unsafe_skce_eval_targets(
    κtargets::Kernel,
    p::AbstractVector{<:Real},
    y::Integer,
    p̃::AbstractVector{<:Real},
    ỹ::Integer
)
    # ensure that y ≤ ỹ (simplifies the implementation)
    y > ỹ && return unsafe_skce_eval_targets(κtargets, p̃, ỹ, p, y)

    # precomputations
    n = length(p)

    @inbounds begin
        py = p[y]
        pỹ = p[ỹ]
        p̃y = p̃[y]
        p̃ỹ = p̃[ỹ]
    end
    pym1 = py - 1
    pỹm1 = pỹ - 1
    p̃ym1 = p̃y - 1
    p̃ỹm1 = p̃ỹ - 1

    # i = y, j = ỹ
    result = κtargets(y, ỹ) * (1 - py - p̃ỹ + py * p̃ỹ)

    # i < y
    for i in 1:(y - 1)
        @inbounds pi = p[i]
        @inbounds p̃i = p̃[i]

        # i = j < y ≤ ỹ
        result += κtargets(i, i) * pi * p̃i

        # i < j < y ≤ ỹ
        @inbounds for j in (i + 1):(y - 1)
            result += κtargets(i, j) * (pi * p̃[j] + p[j] * p̃i)
        end

        # i < y < j < ỹ
        @inbounds for j in (y + 1):(ỹ - 1)
            result += κtargets(i, j) * (pi * p̃[j] + p[j] * p̃i)
        end

        # i < y ≤ ỹ < j
        @inbounds for j in (ỹ + 1):n
            result += κtargets(i, j) * (pi * p̃[j] + p[j] * p̃i)
        end
    end

    # y < i < ỹ
    for i in (y + 1):(ỹ - 1)
        @inbounds pi = p[i]
        @inbounds p̃i = p̃[i]

        # y < i = j < ỹ
        result += κtargets(i, i) * pi * p̃i

        # y < i < j < ỹ
        @inbounds for j in (i + 1):(ỹ - 1)
            result += κtargets(i, j) * (pi * p̃[j] + p[j] * p̃i)
        end

        # y < i < ỹ < j
        @inbounds for j in (ỹ + 1):n
            result += κtargets(i, j) * (pi * p̃[j] + p[j] * p̃i)
        end
    end

    # ỹ < i
    for i in (ỹ + 1):n
        @inbounds pi = p[i]
        @inbounds p̃i = p̃[i]

        # ỹ < i = j
        result += κtargets(i, i) * pi * p̃i

        # ỹ < i < j
        @inbounds for j in (i + 1):n
            result += κtargets(i, j) * (pi * p̃[j] + p[j] * p̃i)
        end
    end

    # handle special case y = ỹ
    if y == ỹ
        # i < y = ỹ, j = y = ỹ
        @inbounds for i in 1:(y - 1)
            result += κtargets(i, y) * (p[i] * p̃ym1 + pym1 * p̃[i])
        end

        # i = y = ỹ, j > y = ỹ
        @inbounds for j in (y + 1):n
            result += κtargets(y, j) * (pym1 * p̃[j] + p[j] * p̃ym1)
        end
    else
        # i < y
        for i in 1:(y - 1)
            @inbounds pi = p[i]
            @inbounds p̃i = p̃[i]

            # j = y < ỹ
            result += κtargets(i, y) * (pi * p̃y + pym1 * p̃i)

            # y < j = ỹ
            result += κtargets(i, ỹ) * (pi * p̃ỹm1 + pỹ * p̃i)
        end

        # i = y = j < ỹ
        result += κtargets(y, y) * pym1 * p̃y

        # i = y < j < ỹ and y < i < j = ỹ
        for ij in (y + 1):(ỹ - 1)
            @inbounds pij = p[ij]
            @inbounds p̃ij = p̃[ij]

            # i = y < j < ỹ
            result += κtargets(y, ij) * (pym1 * p̃ij + pij * p̃y)

            # y < i < j = ỹ
            result += κtargets(ij, ỹ) * (pij * p̃ỹm1 + pỹ * p̃ij)
        end

        # i = ỹ = j
        result += κtargets(ỹ, ỹ) * pỹ * (p̃ỹ - 1)

        # i = y < ỹ < j and i = ỹ < j
        for j in (ỹ + 1):n
            @inbounds pj = p[j]
            @inbounds p̃j = p̃[j]

            # i = y < ỹ < j
            result += κtargets(y, j) * (pym1 * p̃j + pj * p̃y)

            # i = ỹ < j
            result += κtargets(ỹ, j) * (p̃ỹm1 * pj + p̃j * pỹ)
        end
    end

    return result
end

function unsafe_skce_eval_targets(
    ::WhiteKernel,
    p::AbstractVector{<:Real},
    y::Integer,
    p̃::AbstractVector{<:Real},
    ỹ::Integer
)
    @inbounds res = (y == ỹ) - p[ỹ] - p̃[y] + dot(p, p̃)
    return res
end
