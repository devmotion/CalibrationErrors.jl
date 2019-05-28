module CalibrationErrors

using DataStructures
using Distances
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

# scalar-valued kernels
export SquaredExponentialKernel, ExponentialKernel, GaussianKernel, LaplacianKernel

# matrix-valued kernels
export UniformScalingKernel, DiagonalKernel

include("generic.jl")
include("binning/generic.jl")
include("binning/uniform.jl")
include("binning/medianvariance.jl")
include("ece.jl")
include("kernels/generic.jl")
include("kernels/scalar.jl")
include("kernels/matrix.jl")
include("skce/generic.jl")
include("skce/biased.jl")
include("skce/unbiased.jl")

end # module
