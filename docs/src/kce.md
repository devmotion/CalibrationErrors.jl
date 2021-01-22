# [Kernel calibration error (KCE)](@id kce)

## Definition

The kernel calibration error (KCE) is another calibration error. It is based on real-valued
kernels on the product space $\mathcal{P} \times \mathcal{Y}$ of predictions and targets.


The KCE with respect to a real-valued kernel
$k \colon (\mathcal{P} \times \mathcal{Y}) \times (\mathcal{P} \times \mathcal{Y}) \to \mathbb{R}$
is defined[^WLZ21] as
```math
\mathrm{KCE}_k := \sup_{f \in \mathcal{B}_k} \bigg| \mathbb{E}_{Y,P_X} f(P_X, Y) - \mathbb{E}_{Z_X,P_X} f(P_X, Z_X)\bigg|,
```
where $\mathcal{B}_{k}$ is the unit ball in the
[reproducing kernel Hilbert space (RKHS)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space)
to $k$ and $Z_X$ is an artificial random variable on the target space $\mathcal{Y}$ whose
conditional law is given by
```math
Z_X \,|\, P_X = \mu \sim \mu.
```
The RKHS to kernel $k$, and hence also the unit ball $\mathcal{B}_k$, consists of
real-valued functions of the form $f \colon \mathcal{P} \times \mathcal{Y} \to \mathbb{R}$.

For classification models with $m$ classes, there exists an equivalent formulation of the
KCE based on matrix-valued kernel
$\tilde{k} \colon \mathcal{P} \times \mathcal{P} \to \mathbb{R}^{m \times m}$ on
the space $\mathcal{P}$ of predictions.[^WLZ19] The definition above can be rewritten as
```math
\mathrm{KCE}_{\tilde{k}} := \sup_{f \in \mathcal{B}_{\tilde{k}}} \bigg| \mathbb{E}_{P_X} \big(\mathcal{L}(Y \,|\, P_X) - P_X\big)^\mathsf{T} f(P_X) \bigg|,
```
where the matrix-valued kernel $\tilde{k}$ is given by
```math
\tilde{k}_{i,j}(p, q) = k((p, i), (q, j)) \quad (i,j=1,\ldots,m),
```
and $\mathcal{B}_{\tilde{k}}$ is the unit ball in the RKHS of $\tilde{k}$, consisting
of vector-valued functions $f \colon \mathcal{P} \to \mathbb{R}^m$. However,
this formulation applies only to classification models whereas the general
definition above covers all probabilistic predictive models.

For a large class of kernels the KCE is zero if and only if the model is
calibrated.[^WLZ21] Moreover, the squared KCE (SKCE) can be formulated in
terms of the kernel $k$ as
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

The KCE is actually a special case of calibration errors that are formulated as integral
probability metrics of the form
```math
\sup_{f \in \mathcal{F}} \big| \mathbb{E}_{Y,P_X} f(P_X, Y) - \mathbb{E}_{Z_X,P_X} f(P_X, Z_X)\big|,
```
where $\mathcal{F}$ is a space of real-valued functions of the form
$f \colon \mathcal{P} \times \mathcal{Y} \to \mathbb{R}$.[^WLZ21] For classification models,
the [ECE](@ref ece) with respect to common distances such as the total variation distance
or the squared Euclidean distance can be formulated in this way.[^WLZ19]

The maximum mean calibration error (MMCE)[^KSJ] can be viewed as a special case of the KCE, in
which only the most-confident predictions are considered.[^WLZ19]

[^KSJ]: Kumar, A., Sarawagi, S., & Jain, U. (2018). [Trainable calibration measures for neural networks from kernel mean embeddings](http://proceedings.mlr.press/v80/kumar18a.html). In *Proceedings of the 35th International Conference on Machine Learning* (pp. 2805-2814).

[^WLZ19]: Widmann, D., Lindsten, F., & Zachariah, D. (2019). [Calibration tests in multi-class classification: A unifying framework](https://proceedings.neurips.cc/paper/2019/hash/1c336b8080f82bcc2cd2499b4c57261d-Abstract.html). In *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)* (pp. 12257–12267).

[^WLZ21]: Widmann, D., Lindsten, F., & Zachariah, D. (2021). [Calibration tests beyond classification](https://openreview.net/forum?id=-bxf89v3Nx). To be presented at *ICLR 2021*.

## Estimators

For the SKCE biased and unbiased estimators exist. In CalibrationErrors.jl
three types of estimators are available, namely [`BiasedSKCE`](@ref),
[`UnbiasedSKCE`](@ref), and [`BlockUnbiasedSKCE`](@ref). Unsurprisingly,
[`BiasedSKCE`](@ref) is a biased estimator whereas the other two
estimators are unbiased. [`BiasedSKCE`](@ref) and [`UnbiasedSKCE`](@ref)
have quadratic sample complexity whereas [`BlockUnbiasedSKCE`](@ref)
is an estimator with linear sample complexity.

### Biased estimator

```@docs
BiasedSKCE
```

### Unbiased estimators

```@docs
UnbiasedSKCE
```

```@docs
BlockUnbiasedSKCE
```
