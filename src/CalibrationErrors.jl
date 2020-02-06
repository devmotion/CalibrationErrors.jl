module CalibrationErrors

using Reexport

using DataStructures
using Distances
@reexport using KernelFunctions
using Parameters
using StatsBase

using LinearAlgebra
using Statistics

# estimation
export calibrationerror

# estimators
export ECE, BiasedSKCE, LinearUnbiasedSKCE, QuadraticUnbiasedSKCE

# binning algorithms
export UniformBinning, MedianVarianceBinning

# kernels
export TensorProductKernel

include("generic.jl")
include("binning/generic.jl")
include("binning/uniform.jl")
include("binning/medianvariance.jl")
include("ece.jl")

include("kernels.jl")
include("skce/generic.jl")
include("skce/biased.jl")
include("skce/unbiased.jl")

end # module
