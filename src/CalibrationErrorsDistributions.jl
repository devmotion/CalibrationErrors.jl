module CalibrationErrorsDistributions

using Reexport

@reexport using CalibrationErrors
@reexport using Distributions
@reexport using KernelFunctions

using Distances: Distances, Euclidean, SqEuclidean
using ExactOptimalTransport: ExactOptimalTransport
using PDMats: PDMats
using Tulip: Tulip

using LinearAlgebra: LinearAlgebra

const OT = ExactOptimalTransport

include("distances/types.jl")
include("distances/wasserstein.jl")

include("normal.jl")
include("laplace.jl")
include("mvnormal.jl")
include("mixturemodel.jl")

include("deprecated.jl")

end
