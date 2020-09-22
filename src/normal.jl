# predicted normal distributions with squared exponential kernel for the targets
function CalibrationErrors.unsafe_skce_eval_targets(
    ::SqExponentialKernel,
    p::Normal,
    y::Real,
    p̃::Normal,
    ỹ::Real
)
    return unsafe_skce_eval_targets_normal_gaussian(params(p), y, params(p̃), ỹ)
end

function CalibrationErrors.unsafe_skce_eval_targets(
    κtargets::TransformedKernel{<:SqExponentialKernel,<:ScaleTransform},
    p::Normal,
    y::Real,
    p̃::Normal,
    ỹ::Real
)
    # obtain the parameters of the predicted Laplace distributions
    μ, σ = params(p)
    μ̃, σ̃ = params(p̃)

    # obtain scale parameter
    invν = first(κtargets.transform.s)

    return unsafe_skce_eval_targets_normal_gaussian(
        (invν * μ, invν * σ), invν * y, (invν * μ̃, invν * σ̃), invν * ỹ
    )
end

function unsafe_skce_eval_targets_normal_gaussian((μ, σ), y, (μ̃, σ̃), ỹ)
    # compute variances
    σ2 = σ^2
    σ̃2 = σ̃^2

    # compute scaling factors
    α = inv(1 + σ2)
    β = inv(1 + σ̃2)
    γ = inv(1 + σ2 + σ̃2)

    ky = exp(- (y - ỹ)^2 / 2) - sqrt(α) * exp(- α * (μ - ỹ)^2 / 2) -
        sqrt(β) * exp(- β * (y - μ̃)^2 / 2) + sqrt(γ) * exp(- γ * (μ - μ̃)^2 / 2)

    return ky
end

function CalibrationErrors.unsafe_ucme_eval_targets(
    κtargets::SqExponentialKernel,
    p::Normal,
    y::Real,
    testp::Normal,
    testy::Real
)
    # extract parameters
    μ, σ = params(p)

    # compute scaling factor
    α = inv(1 + σ^2)

    return exp(- (y - testy)^2 / 2) - sqrt(α) * exp(- α * (μ - testy)^2 / 2)
end
