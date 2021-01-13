# [Kernel calibration error (KCE)](@id kce)

## Definition

The kernel calibration error (KCE) is another calibration error. The KCE with
respect to a real-valued kernel
$k \colon (\mathcal{P} \times \mathcal{Y}) \times (\mathcal{P} \times \mathcal{Y}) \to \mathbb{R}$
on the product space $\mathcal{P} \times \mathcal{Y}$ of predictions and targets
is defined as
as
```math
\mathrm{KCE}_k := \sup_{f \in \mathcal{B}_k} \bigg| \mathbb{E}_{Y,P_X} f(P_X, Y) - \mathbb{E}_{Z_X,P_X} f(P_X, Z_X)\bigg|,
```
where $\mathcal{B}_{k}$ is the unit ball in the
[reproducing kernel Hilbert space (RKHS)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space)
of $k$, consisting of functions $f \colon \mathcal{P} \times \mathcal{Y} \to \mathbb{R}$,
and $Z_X$ is a random variable on the target space $\mathcal{Y}$ with conditional distribution
$P_X$ given $P_X$, i.e., $Z_X \,|\, P_X = \mu \sim \mu$
([Widmann et al., 2021](https://openreview.net/pdf?id=-bxf89v3Nx)).[^1]

For classification models with $m$ classes, there exists an equivalent formulation of the
KCE based on matrix-valued kernel
$\tilde{k} \colon \mathcal{P} \times \mathcal{P} \to \mathbb{R}^{m \times m}$ on
the space $\mathcal{P}$ of predictions
([Widmann et al., 2019](https://proceedings.neurips.cc/paper/2019/file/1c336b8080f82bcc2cd2499b4c57261d-Paper.pdf)). The definition above can be rewritten as
```math
\mathrm{KCE}_{\tilde{k}} := \sup_{f \in \mathcal{B}_{\tilde{k}}} \bigg| \mathbb{E}_{P_X} \big(\mathcal{L}(Y \,|\, P_X) - P_X\big)^\mathsf{T} f(P_X) \bigg|,
```
where the matrix-valued kernel $\tilde{k}$ is given by
```math
\tilde{k}_{i,j}(p, q) = k((p, i), (q, j)) \quad (i,j=1,\ldots,m),
```
and $\mathcal{B}_{\tilde{k}}$ is the unit ball in the RKHS of $\tilde{k}$, consisting
of vector-valued functions $f \colon \mathcal{P} \to \mathbb{R}^m$.[^2] However,
this formulation applies only to classification models whereas the general
definition above covers all probabilistic predictive models.

For a large class of kernels the KCE is zero if and only if the model is
calibrated ([Widmann et al., 2021](https://openreview.net/pdf?id=-bxf89v3Nx)).
Moreover, the squared KCE (SKCE) can be formulated in terms of the kernel $k$ as
```math
\begin{aligned}
\mathrm{SKCE}_{k} := \mathrm{KCE}_k^2 &= \int k(u, v) \, \big(\mathcal{L}(P_X, Y) - \mathcal{L}(P_X, Z_X)\big)(u) \big(\mathcal{L}(P_X, Y) - \mathcal{L}(P_X, Z_X)\big)(v) \\
&= \mathbb{E} h_k\big((P_X, Y), (P_{X'}, Y')\big),
\end{aligned}
```
where $(X',Y')$ is an independent copy of $(X,Y)$ and
```math
\begin{aligned}
h_k\big((\mu, y), (\mu', y')\big) :={}& k\big((\mu, y), (\mu', y')\big) - \mathbb{E}_{Z \sim \mu} k\big((\mu, Z), (\mu', y')\big) \\
&- \mathbb{E}_{Z' \sim \mu'} k\big((\mu, y), (\mu', Z')\big) + \mathbb{E}_{Z \sim \mu, Z' \sim \mu'} k\big((\mu, Z), (\mu', Z')\big).
\end{aligned}
```

[^1]: The KCE is a special case of calibration errors of the form $\sup_{f \in \mathcal{F}} \big| \mathbb{E}_{Y,P_X} f(P_X, Y) - \mathbb{E}_{Z_X,P_X} f(P_X, Z_X)\big|$, where $\mathcal{F}$ is some space of functions $f \colon \mathcal{P} \times \mathcal{Y} \to \mathbb{R}$ ([Widmann et al., 2021](https://openreview.net/pdf?id=-bxf89v3Nx)). For classification models, the [ECE](@ref ece) with respect to common distances such as the total variation distance or the squared Euclidean distance can be formulated in this way ([Widmann et al., 2019](https://proceedings.neurips.cc/paper/2019/file/1c336b8080f82bcc2cd2499b4c57261d-Paper.pdf)).

[^2]: The maximum mean calibration error (MMCE) ([Kumar et al., 2018](http://proceedings.mlr.press/v80/kumar18a/kumar18a.pdf)) can be viewed as a special case of the KCE, in which only the most-confident predictions are considered ([Widmann et al., 2019](https://proceedings.neurips.cc/paper/2019/file/1c336b8080f82bcc2cd2499b4c57261d-Paper.pdf)).

## Estimators

For the SKCE biased and unbiased estimators exist. In CalibrationErrors.jl
three types of estimators are available, namely [`BiasedSKCE`](@ref),
[`UnbiasedSKCE`](@ref), and [`BlockUnbiasedSKCE`](@ref). Unsurprisingly,
[`BiasedSKCE`](@ref) is a biased estimator whereas the other two
estimators are unbiased. [`BiasedSKCE`](@ref) and [`UnbiasedSKCE`](@ref)
have quadratic sample complexity whereas [`BlockUnbiasedSKCE`](@ref)
is an estimator with linear sample complexity.

### Biased estimators

```@docs
BiasedSKCE
BiasedSKCE(::Kernel, ::Kernel)
```

### Unbiased estimators

```@docs
UnbiasedSKCE
UnbiasedSKCE(::Kernel, ::Kernel)
```

```@docs
BlockUnbiasedSKCE
BlockUnbiasedSKCE(::Kernel, ::Kernel)
```
