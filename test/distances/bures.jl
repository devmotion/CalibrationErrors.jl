using CalibrationErrorsDistributions
using PDMats

using LinearAlgebra
using Test

@testset "bures.jl" begin
    function _sqbures(A, B)
        sqrt_A = sqrt(A)
        return tr(A) + tr(B) - 2 * tr(sqrt(sqrt_A * B * sqrt_A'))
    end

    function rand_matrices(n)
        A = randn(n, n)
        B = A' * A + I
        return B, PDMat(B), PDiagMat(diag(B)), ScalMat(n, B[1])
    end

    for (x, y) in Iterators.product(rand_matrices(10), rand_matrices(10))
        xfull = Matrix(x)
        yfull = Matrix(y)
        @test CalibrationErrorsDistributions.sqbures(x, y) ≈ _sqbures(xfull, yfull)
    end
end