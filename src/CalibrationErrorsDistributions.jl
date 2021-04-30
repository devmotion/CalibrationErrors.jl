module CalibrationErrorsDistributions

using Reexport

@reexport using CalibrationErrors
@reexport using Distributions
@reexport using KernelFunctions

using Distances: Distances
using MathOptInterface: MathOptInterface
using PDMats: PDMats
using Tulip: Tulip

using LinearAlgebra: LinearAlgebra

export WassersteinExponentialKernel, MixtureWassersteinExponentialKernel

const MOI = MathOptInterface

include("optimaltransport.jl")

include("distances/types.jl")
include("distances/bures.jl")
include("distances/wasserstein.jl")

include("kernels.jl")

include("normal.jl")
include("laplace.jl")
include("mvnormal.jl")
include("mixturemodel.jl")

end
