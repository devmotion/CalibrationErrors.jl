# SKCE

function CalibrationErrors.unsafe_skce_eval_targets(
    kernel::SqExponentialKernel{Euclidean},
    p::MvNormal,
    y::AbstractVector{<:Real},
    p̃::MvNormal,
    ỹ::AbstractVector{<:Real},
)
    # extract the parameters
    μ = p.μ
    Σ = p.Σ
    μ̃ = p̃.μ
    Σ̃ = p̃.Σ

    # compute inverse scaling matrices
    invA = LinearAlgebra.I + Σ
    invB = LinearAlgebra.I + Σ̃
    invC = invA + Σ̃

    res =
        kernel(y, ỹ) - mvnormal_gaussian_kernel(invA, μ, ỹ) -
        mvnormal_gaussian_kernel(invB, μ̃, y) + mvnormal_gaussian_kernel(invC, μ, μ̃)

    return res
end

function mvnormal_gaussian_kernel(invA, y, ỹ)
    return exp(-(LinearAlgebra.logdet(invA) + invquad_diff(invA, y, ỹ)) / 2)
end

# UCME

# predicted normal distributions with squared exponential kernel for the targets
function CalibrationErrors.unsafe_ucme_eval_targets(
    kernel::SqExponentialKernel{Euclidean},
    p::MvNormal,
    y::AbstractVector{<:Real},
    testp::MvNormal,
    testy::AbstractVector{<:Real},
)
    # compute inverse scaling matrix
    invA = LinearAlgebra.I + p.Σ

    return kernel(y, testy) - mvnormal_gaussian_kernel(invA, p.μ, testy)
end

# kernels with input transformations
# TODO: scale upfront?

function CalibrationErrors.unsafe_skce_eval_targets(
    kernel::TransformedKernel{
        SqExponentialKernel{Euclidean},<:Union{ScaleTransform,ARDTransform,LinearTransform}
    },
    p::MvNormal,
    y::AbstractVector{<:Real},
    p̃::MvNormal,
    ỹ::AbstractVector{<:Real},
)
    # obtain the transform
    t = kernel.transform

    return CalibrationErrors.unsafe_skce_eval_targets(
        SqExponentialKernel(), apply(t, p), t(y), apply(t, p̃), t(ỹ)
    )
end

function CalibrationErrors.unsafe_ucme_eval_targets(
    kernel::TransformedKernel{
        SqExponentialKernel{Euclidean},<:Union{ScaleTransform,ARDTransform,LinearTransform}
    },
    p::MvNormal,
    y::AbstractVector{<:Real},
    testp::MvNormal,
    testy::AbstractVector{<:Real},
)
    # obtain the transform
    t = kernel.transform

    # `testp` is irrelevant for the evaluation and therefore not transformed
    return CalibrationErrors.unsafe_ucme_eval_targets(
        SqExponentialKernel(), apply(t, p), t(y), testp, t(testy)
    )
end

## utilities

# internal `apply` avoids type piracy of transforms
function apply(t::ScaleTransform, d::MvNormal)
    s = first(t.s)
    return MvNormal(t(d.μ), s^2 * d.Σ)
end
function apply(t::ARDTransform, d::MvNormal)
    return MvNormal(t(d.μ), scale_cov(d.Σ, t.v))
end
# `X_A_Xt` only works with StridedMatrix: https://github.com/JuliaStats/PDMats.jl/issues/96
apply(t::LinearTransform, d::MvNormal) = Matrix(t.A) * d

# scale covariance matrix
# TODO: improve efficiency
scale_cov(Σ::PDMats.ScalMat, v) = PDMats.PDiagMat(v .^ 2 .* Σ.value)
scale_cov(Σ::PDMats.PDiagMat, v) = PDMats.PDiagMat(v .^ 2 .* Σ.diag)
scale_cov(Σ::PDMats.AbstractPDMat, v) = PDMats.X_A_Xt(Σ, LinearAlgebra.diagm(v))

invquad_diff(A::PDMats.ScalMat, x, y) = A.inv_value * Distances.sqeuclidean(x, y)
invquad_diff(A::PDMats.PDiagMat, x, y) = Distances.wsqeuclidean(x, y, A.inv_diag)
invquad_diff(A::PDMats.AbstractPDMat, x, y) = PDMats.invquad(A, x .- y)
