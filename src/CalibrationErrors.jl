module CalibrationErrors

using Reexport

using DataStructures
@reexport using Distances
@reexport using Distributions
using ExactOptimalTransport: ExactOptimalTransport
@reexport using KernelFunctions
using PDMats: PDMats
using StatsBase: StatsBase
using Tulip: Tulip
using UnPack: @unpack

using LinearAlgebra
using Statistics

const OT = ExactOptimalTransport

# estimation
export calibrationerror

# estimators
export ECE, SKCE, UCME

# binning algorithms
export UniformBinning, MedianVarianceBinning

# Wasserstein distances
export Wasserstein, SqWasserstein, MixtureWasserstein, SqMixtureWasserstein

include("distances/types.jl")
include("distances/wasserstein.jl")

include("generic.jl")
include("binning/generic.jl")
include("binning/uniform.jl")
include("binning/medianvariance.jl")
include("ece.jl")

include("skce.jl")

include("ucme.jl")

include("distributions/normal.jl")
include("distributions/laplace.jl")
include("distributions/mvnormal.jl")
include("distributions/mixturemodel.jl")

end # module
