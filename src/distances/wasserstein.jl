struct SqWasserstein <: DistributionsSemiMetric end

# result type (e.g., for pairwise computations)
function Distances.result_type(
    ::SqWasserstein, ::Type{T1}, ::Type{T2}
) where {T1<:Real,T2<:Real}
    return promote_type(T1, T2)
end

# evaluations for normal distributions
function (::SqWasserstein)(a::Normal, b::Normal)
    μa, σa = params(a)
    μb, σb = params(b)
    return abs2(μa - μb) + abs2(σa - σb)
end

function (::SqWasserstein)(a::AbstractMvNormal, b::AbstractMvNormal)
    μ1 = mean(a)
    μ2 = mean(b)
    Σ1 = cov(a)
    Σ2 = cov(b)
    return Distances.sqeuclidean(μ1, μ2) + OT.sqbures(Σ1, Σ2)
end

function (::SqWasserstein)(a::MvNormal, b::MvNormal)
    μa, Σa = params(a)
    μb, Σb = params(b)
    return Distances.sqeuclidean(μa, μb) + OT.sqbures(Σa, Σb)
end

# evaluations for Laplace distributions
function (::SqWasserstein)(a::Laplace, b::Laplace)
    μa, βa = params(a)
    μb, βb = params(b)
    return abs2(μa - μb) + 2 * abs2(βa - βb)
end

# Wasserstein 2 distance
struct Wasserstein <: DistributionsMetric end

# result type (e.g., for pairwise computations)
function Distances.result_type(
    ::Wasserstein, ::Type{T1}, ::Type{T2}
) where {T1<:Real,T2<:Real}
    return float(promote_type(T1, T2))
end

function (::Wasserstein)(a::Distribution, b::Distribution)
    return sqrt(SqWasserstein()(a, b))
end

# Mixture Wasserstein distances
struct SqMixtureWasserstein{S} <: DistributionsSemiMetric
    lpsolver::S
end
struct MixtureWasserstein{S} <: DistributionsMetric
    lpsolver::S
end
SqMixtureWasserstein() = SqMixtureWasserstein(Tulip.Optimizer())
MixtureWasserstein() = MixtureWasserstein(Tulip.Optimizer())

# result type (e.g., for pairwise computations)
function Distances.result_type(
    ::SqMixtureWasserstein, ::Type{T1}, ::Type{T2}
) where {T1<:Real,T2<:Real}
    return promote_type(T1, T2)
end
function Distances.result_type(
    ::MixtureWasserstein, ::Type{T1}, ::Type{T2}
) where {T1<:Real,T2<:Real}
    return float(promote_type(T1, T2))
end

function (s::SqMixtureWasserstein)(a::AbstractMixtureModel, b::AbstractMixtureModel)
    C = Distances.pairwise(SqWasserstein(), components(a), components(b))
    return OT.emd2(probs(a), probs(b), C, deepcopy(s.lpsolver))
end

function (m::MixtureWasserstein)(a::AbstractMixtureModel, b::AbstractMixtureModel)
    return sqrt(SqMixtureWasserstein(m.lpsolver)(a, b))
end
