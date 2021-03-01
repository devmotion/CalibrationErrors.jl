# SKCE

# predicted Laplace distributions with exponential kernel for the targets
function CalibrationErrors.unsafe_skce_eval_targets(
    kernel::ExponentialKernel, p::Laplace, y::Real, p̃::Laplace, ỹ::Real
)
    # extract the parameters
    μ = p.μ
    β = p.θ
    μ̃ = p̃.μ
    β̃ = p̃.θ

    res =
        kernel(y, ỹ) - laplace_laplacian_kernel(β, abs(μ - ỹ)) -
        laplace_laplacian_kernel(β̃, abs(μ̃ - y)) +
        laplace_laplacian_kernel(β, β̃, abs(μ - μ̃))

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
            return β̃^3 * exp(-z / β̃) / csq - ((1 + z) / (2 * c) + β̃^2 / csq) * exp(-z)
        end
    end

    if isone(β̃)
        c = β^2 - 1
        csq = c^2
        return β^3 * exp(-z / β) / csq - ((1 + z) / (2 * c) + β^2 / csq) * exp(-z)
    elseif β̃ == β
        c = β^2 - 1
        csq = c^2
        return exp(-z) / csq + ((β + z) / (2 * c) - β / csq) * exp(-z / β)
    else
        c1 = β^2 - 1
        c2 = β̃^2 - 1
        c3 = β^2 - β̃^2
        return β^3 * exp(-z / β) / (c1 * c3) - β̃^3 * exp(-z / β̃) / (c2 * c3) +
               exp(-z) / (c1 * c2)
    end
end

# UCME

function CalibrationErrors.unsafe_ucme_eval_targets(
    kernel::ExponentialKernel, p::Laplace, y::Real, ::Laplace, testy::Real
)
    return kernel(y, testy) - laplace_laplacian_kernel(p.θ, abs(p.μ - testy))
end

# kernels with input transformations
# TODO: scale upfront?
function CalibrationErrors.unsafe_skce_eval_targets(
    kernel::TransformedKernel{ExponentialKernel,<:Union{ScaleTransform,ARDTransform}},
    p::Laplace,
    y::Real,
    p̃::Laplace,
    ỹ::Real,
)
    # obtain the transform
    t = kernel.transform

    return CalibrationErrors.unsafe_skce_eval_targets(
        ExponentialKernel(), apply(t, p), t(y), apply(t, p̃), t(ỹ)
    )
end

function CalibrationErrors.unsafe_ucme_eval_targets(
    kernel::TransformedKernel{ExponentialKernel,<:Union{ScaleTransform,ARDTransform}},
    p::Laplace,
    y::Real,
    testp::Laplace,
    testy::Real,
)
    # obtain the transform
    t = kernel.transform

    # `testp` is irrelevant for the evaluation and therefore not transformed
    return CalibrationErrors.unsafe_ucme_eval_targets(
        ExponentialKernel(), apply(t, p), t(y), testp, t(testy)
    )
end

# utilities

# internal `apply` avoids type piracy of transforms
apply(t::Union{ScaleTransform,ARDTransform}, d::Laplace) = Laplace(t(d.μ), t(d.θ))
