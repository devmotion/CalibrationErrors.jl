# abstract type for matrix-valued kernels
abstract type MatrixKernel <: AbstractKernel end

# uniform scaling of scalar-valued kernel
struct UniformScalingKernel{T<:Real,K} <: MatrixKernel
    λ::T
    kernel::K

    function UniformScalingKernel{T,K}(λ::T, kernel::K) where {T,K}
        λ ≥ zero(λ) || throw(ArgumentError("`λ` must be non-negative"))

        new{T,K}(λ, kernel)
    end
end

UniformScalingKernel(λ::Real, kernel) =
    UniformScalingKernel{typeof(λ),typeof(kernel)}(λ, kernel)
UniformScalingKernel(kernel) = UniformScalingKernel(1, kernel)

(kernel::UniformScalingKernel)(x, y) = UniformScaling(kernel.λ * kernel.kernel(x, y))

kernel_result_type(kernel::UniformScalingKernel, x, y) =
    typeof(zero(kernel.λ) * kernel.kernel(zero(eltype(x)), zero(eltype(y))))

# scaling of scalar-valued kernel by diagonal matrix
struct DiagonalKernel{V<:AbstractVector{<:Real},K} <: MatrixKernel
    diag::V
    kernel::K

    function DiagonalKernel{V,K}(diag::V, kernel::K) where {V,K}
        all(x -> x ≥ zero(x), diag) ||
            throw(ArgumentError("all entries of `diag` must be non-negative"))

        new{V,K}(diag, kernel)
    end
end

DiagonalKernel(diag::AbstractVector{<:Real}, kernel) =
    DiagonalKernel{typeof(diag),typeof(kernel)}(diag, kernel)

(kernel::DiagonalKernel)(x, y) = Diagonal(kernel.diag .* kernel.kernel(x, y))

kernel_result_type(kernel::DiagonalKernel, x, y) =
    typeof(zero(eltype(kernel.diag)) * kernel.kernel(zero(eltype(x)), zero(eltype(y))))
