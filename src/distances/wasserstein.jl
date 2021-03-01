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
    return Distances.sqeuclidean(μ1, μ2) + sqbures(Σ1, Σ2)
end

function (::SqWasserstein)(a::MvNormal, b::MvNormal)
    μa, Σa = params(a)
    μb, Σb = params(b)
    return Distances.sqeuclidean(μa, μb) + sqbures(Σa, Σb)
end

# evaluations for Laplace distributions
function (::SqWasserstein)(a::Laplace, b::Laplace)
    μa, βa = params(a)
    μb, βb = params(b)
    return abs2(μa - μb) + 2 * abs2(βa - βb)
end

# syntactic sugar
function sqwasserstein(a::Distribution, b::Distribution)
    return (SqWasserstein())(a, b)
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
    return sqrt(sqwasserstein(a, b))
end

function wasserstein(a::Distribution, b::Distribution)
    return (Wasserstein())(a, b)
end

# Mixture Wasserstein distances
struct SqMixtureWasserstein <: DistributionsSemiMetric end
struct MixtureWasserstein <: DistributionsMetric end

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

function (::SqMixtureWasserstein)(a::AbstractMixtureModel, b::AbstractMixtureModel)
    probsa = probs(a)
    componentsa = components(a)
    probsb = probs(b)
    componentsb = components(b)

    C = Distances.pairwise(SqWasserstein(), componentsa, componentsb)
    return OptimalTransport.emd2(probsa, probsb, C)
end

function sqmixturewasserstein(a::AbstractMixtureModel, b::AbstractMixtureModel)
    return (SqMixtureWasserstein())(a, b)
end

function (::MixtureWasserstein)(a::AbstractMixtureModel, b::AbstractMixtureModel)
    return sqrt(sqmixturewasserstein(a, b))
end

function mixturewasserstein(a::AbstractMixtureModel, b::AbstractMixtureModel)
    return (MixtureWasserstein())(a, b)
end
