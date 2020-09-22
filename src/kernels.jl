struct WassersteinExponentialKernel <: KernelFunctions.SimpleKernel end

KernelFunctions.kappa(κ::WassersteinExponentialKernel, d) = exp(-d)
KernelFunctions.metric(::WassersteinExponentialKernel) = Wasserstein()

# Wasserstein-like exponential kernel for mixture models
struct MixtureWassersteinExponentialKernel <: KernelFunctions.SimpleKernel end

KernelFunctions.kappa(κ::MixtureWassersteinExponentialKernel, d) = exp(-d)
KernelFunctions.metric(::MixtureWassersteinExponentialKernel) = MixtureWasserstein()

# fast implementation for length scales
for T in (WassersteinExponentialKernel, MixtureWassersteinExponentialKernel)
    @eval begin
        function (k::TransformedKernel{$T,<:ScaleTransform})(
            x::Distribution,
            y::Distribution,
        )
            d = Distances.evaluate(KernelFunctions.metric(k.kernel), x, y)
            return KernelFunctions.kappa(k.kernel, first(k.transform.s) * d)
        end
    end
end
