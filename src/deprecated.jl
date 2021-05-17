@deprecate WassersteinExponentialKernel() ExponentialKernel(; metric=Wasserstein())
@deprecate MixtureWassersteinExponentialKernel() ExponentialKernel(;
    metric=MixtureWasserstein()
)
