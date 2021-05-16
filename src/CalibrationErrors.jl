module CalibrationErrors

using Reexport

using DataStructures
using Distances
@reexport using KernelFunctions
using StatsBase
using UnPack

using LinearAlgebra
using Statistics

# estimation
export calibrationerror

# estimators
export ECE, BiasedSKCE, UnbiasedSKCE, BlockUnbiasedSKCE, UCME

# binning algorithms
export UniformBinning, MedianVarianceBinning

include("generic.jl")
include("binning/generic.jl")
include("binning/uniform.jl")
include("binning/medianvariance.jl")
include("ece.jl")

include("skce/generic.jl")
include("skce/biased.jl")
include("skce/unbiased.jl")

include("ucme.jl")

include("deprecated.jl")

end # module
