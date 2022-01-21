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
export ECE, BiasedSKCE, UnbiasedSKCE, BlockUnbiasedSKCE, UCME

# binning algorithms
export UniformBinning, MedianVarianceBinning

include("distances/types.jl")
include("distances/wasserstein.jl")

include("generic.jl")
include("binning/generic.jl")
include("binning/uniform.jl")
include("binning/medianvariance.jl")
include("ece.jl")

include("skce/generic.jl")
include("skce/biased.jl")
include("skce/unbiased.jl")

include("ucme.jl")

include("normal.jl")
include("laplace.jl")
include("mvnormal.jl")
include("mixturemodel.jl")

include("deprecated.jl")

end # module
