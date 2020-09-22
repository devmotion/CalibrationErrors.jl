module CalibrationErrorsDistributions

using Reexport

@reexport using CalibrationErrors
@reexport using Distributions
@reexport using KernelFunctions

import Distances
import OptimalTransport
import PDMats

import LinearAlgebra

export WassersteinExponentialKernel, MixtureWassersteinExponentialKernel

include("distances/types.jl")
include("distances/bures.jl")
include("distances/wasserstein.jl")
include("distances/pairwise.jl")

include("kernels.jl")

include("normal.jl")
include("laplace.jl")
include("mvnormal.jl")
include("mixturemodel.jl")

end
