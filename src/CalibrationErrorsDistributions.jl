module CalibrationErrorsDistributions

using Reexport

@reexport using CalibrationErrors
@reexport using Distributions
@reexport using KernelFunctions

using Distances: Distances
using OptimalTransport: OptimalTransport
using PDMats: PDMats
using Tulip: Tulip

using LinearAlgebra: LinearAlgebra

export WassersteinExponentialKernel, MixtureWassersteinExponentialKernel

include("distances/types.jl")
include("distances/bures.jl")
include("distances/wasserstein.jl")

include("kernels.jl")

include("normal.jl")
include("laplace.jl")
include("mvnormal.jl")
include("mixturemodel.jl")

end
