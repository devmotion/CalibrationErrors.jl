using CalibrationErrorsDistributions
using Distances
using LinearAlgebra
using MathOptInterface
using PDMats
using Random
using Test

using CalibrationErrorsDistributions:
    Wasserstein,
    SqWasserstein,
    MixtureWasserstein,
    SqMixtureWasserstein,
    sqwasserstein,
    optimal_transport_map
using Tulip: Tulip

# only add OptimalTransport on >= Julia 1.4
# Julia 1.3 is only supported by OptimalTransport 0.1 which requires Python
@static if VERSION >= v"1.4"
    using Pkg: Pkg
    Pkg.add(;
        name="OptimalTransport", uuid="7e02d93a-ae51-4f58-b602-d97af76e3b33", version="0.2"
    )
    using OptimalTransport
end

Random.seed!(1234)

const MOI = MathOptInterface

@testset "CalibrationErrorsDistributions" begin
    @testset "optimal transport" begin
        include("optimaltransport.jl")
    end

    @testset "distances" begin
        @testset "Bures metric" begin
            include("distances/bures.jl")
        end
        @testset "Wasserstein distance" begin
            include("distances/wasserstein.jl")
        end
    end

    @testset "kernels" begin
        include("kernels.jl")
    end

    @testset "Normal" begin
        include("normal.jl")
    end
    @testset "MvNormal" begin
        include("mvnormal.jl")
    end
end
