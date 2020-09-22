# predicted Laplace distributions with exponential kernel for the targets
function CalibrationErrors.unsafe_skce_eval_targets(
    ::ExponentialKernel,
    p::Laplace,
    y::Real,
    p̃::Laplace,
    ỹ::Real
)
    return unsafe_skce_eval_targets_laplace_laplacian(params(p), y, params(p̃), ỹ)
end

function CalibrationErrors.unsafe_skce_eval_targets(
    kernel::TransformedKernel{<:ExponentialKernel,<:ScaleTransform},
    p::Laplace,
    y::Real,
    p̃::Laplace,
    ỹ::Real
)
    # obtain the parameters of the predicted Laplace distributions
    μ, β = params(p)
    μ̃, β̃ = params(p̃)

    # obtain inverse length scale
    invν = first(kernel.transform.s)

    return unsafe_skce_eval_targets_laplace_laplacian(
        (invν * μ, invν * β), invν * y, (invν * μ̃, invν * β̃), invν * ỹ
    )
end

function unsafe_skce_eval_targets_laplace_laplacian((μ, β), y, (μ̃, β̃), ỹ)
    res = exp(- (y - ỹ)^2) - laplace_laplacian_kernel(β, abs(μ - ỹ)) -
        laplace_laplacian_kernel(β̃, abs(μ̃ - y)) + laplace_laplacian_kernel(β, β̃, abs(μ - μ̃))

    return res
end

# z = abs(μ - y)
function laplace_laplacian_kernel(β, z)
    if isone(β)
        (1 + z) * exp(-z) / 2
    else
        (β * exp(-z / β) - exp(-z)) / (β^2 - 1)
    end
end

# z = abs(μ - μ̃)
function laplace_laplacian_kernel(β, β̃, z)
    if isone(β)
        if isone(β̃)
            return (3 + 3 * z + z^2) * exp(-z) / 8
        else
            c = β̃^2 - 1
            csq = c^2
            return β̃^3 * exp(- z / β̃) / csq - ((1 + z) / (2 * c) + β̃^2 / csq) * exp(-z)
        end
    end

    if isone(β̃)
        c = β^2 - 1
        csq = c^2
        return β^3 * exp(- z / β) / csq - ((1 + z) / (2 * c) + β^2 / csq) * exp(-z)
    elseif β̃ == β
        c = β^2 - 1
        csq = c^2
        return exp(- z) / csq + ((β + z) / (2 * c) - β / csq) * exp(-z / β)
    else
        c1 = β^2 - 1
        c2 = β̃^2 - 1
        c3 = β^2 - β̃^2
        return β^3 * exp(- z / β) / (c1 * c3) - β̃^3 * exp(- z / β̃) / (c2 * c3) +
            exp(-z) / (c1 * c2)
    end
end