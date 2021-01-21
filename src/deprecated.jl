@deprecate BiasedSKCE(k1::Kernel, k2::Kernel) BiasedSKCE(k1 ⊗ k2)
@deprecate UnbiasedSKCE(k1::Kernel, k2::Kernel) UnbiasedSKCE(k1 ⊗ k2)
@deprecate BlockUnbiasedSKCE(k1::Kernel, k2::Kernel) BlockUnbiasedSKCE(k1 ⊗ k2)
@deprecate BlockUnbiasedSKCE(k1::Kernel, k2::Kernel, blocksize::Int) BlockUnbiasedSKCE(
    k1 ⊗ k2, blocksize,
)
@deprecate UCME(k1::Kernel, k2::Kernel, data...) UCME(k1 ⊗ k2, data...)
