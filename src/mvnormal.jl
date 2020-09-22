function CalibrationErrors.unsafe_skce_eval_targets(
    κ::SqExponentialKernel,
    p::MvNormal,
    y::AbstractVector{<:Real},
    p̃::MvNormal,
    ỹ::AbstractVector{<:Real},
)
    # obtain the parameters of the predicted normal distributions
    μ = mean(p)
    Σ = p.Σ
    μ̃ = mean(p̃)
    Σ̃ = p̃.Σ

    return unsafe_skce_eval_targets_mvnormal_gaussian((μ, Σ), y, (μ̃, Σ̃), ỹ)
end

function unsafe_skce_eval_targets_mvnormal_gaussian((μ, Σ), y, (μ̃, Σ̃), ỹ)
    res = exp(- sqeuclidean(y, ỹ) / 2) - mvnormal_gaussian_kernel(Σ, μ, ỹ) -
        mvnormal_gaussian_kernel(Σ̃, μ̃, y) + mvnormal_gaussian_kernel(Σ + Σ̃, μ, μ̃)

    return res
end

function mvnormal_gaussian_kernel(Σ, μ, ỹ)
    C = Σ + LinearAlgebra.I
    return exp(-logdet(C) / 2 - invquad_diff(C, μ, ỹ))
end

invquad_diff(A::PDMats.ScalMat, x, y) = sqeuclidean(x, y) / A.value
invquad_diff(A::PDMats.PDiagMat, x, y) = wsqeuclidean(inv.(A.diag), x, y)
invquad_diff(A::PDMats.AbstractPDMat, x, y) = PDMats.invquad(A, x .- y)

# predicted normal distributions with squared exponential kernel for the targets
function CalibrationErrors.unsafe_ucme_eval_targets(
    κ::SqExponentialKernel,
    p::MvNormal,
    y::AbstractVector{<:Real},
    testp::MvNormal,
    testy::AbstractVector{<:Real}
)
    # extract parameters
    μ = mean(p)
    Σ = p.Σ

    return exp(- sqeuclidean(y, testy) / 2) - mvnormal_gaussian_kernel(Σ, μ, testy)
end