# abstract type for scalar-valued kernels
abstract type ScalarKernel <: AbstractKernel end

# scalar-valued exponential kernel
struct ExponentialKernel{T<:Real,M<:Metric} <: ScalarKernel
    """Inverse length-scale."""
    γ::T
    """Distance function."""
    distance::M

    function ExponentialKernel{T,M}(γ::T, distance::M) where {T,M}
        γ > zero(γ) ||
            throw(ArgumentError("inverse length-scale has to be positive"))

        new{T,M}(γ, distance)
    end
end

ExponentialKernel(γ::Real, distance::Metric = Euclidean()) =
    ExponentialKernel{typeof(γ),typeof(distance)}(γ, distance)
ExponentialKernel(distance::Metric = Euclidean()) = ExponentialKernel(1, distance)

(kernel::ExponentialKernel)(x, y) = exp(- kernel.γ * evaluate(kernel.distance, x, y))

const LaplacianKernel = ExponentialKernel

# scalar-valued squared exponential kernel
struct SquaredExponentialKernel{T<:Real,M<:Metric} <: ScalarKernel
    """Squared inverse length-scale."""
    γ::T
    """Distance function."""
    distance::M

    function SquaredExponentialKernel{T,M}(γ::T, distance::M) where {T,M}
        γ > zero(γ) ||
            throw(ArgumentError("squared inverse length-scale has to be positive"))

        new{T,M}(γ, distance)
    end
end

SquaredExponentialKernel(γ::Real, distance::Metric = Euclidean()) =
    SquaredExponentialKernel{typeof(γ),typeof(distance)}(γ, distance)
SquaredExponentialKernel(distance::Metric = Euclidean()) =
    SquaredExponentialKernel(1, distance)

(kernel::SquaredExponentialKernel)(x, y) =
    exp(- kernel.γ * evaluate(kernel.distance, x, y)^2)

const GaussianKernel = SquaredExponentialKernel
