# [Expected calibration error (ECE)](@id ece)

## Definition

A common calibration measure is the so-called expected calibration error (ECE).
In its most general form, the ECE with respect to distance measure $d(p, p')$
is defined[^WLZ21] as
```math
\mathrm{ECE}_d := \mathbb{E} d\big(P_X, \mathcal{L}(Y \,|\, P_X)\big).
```

As implied by its name, the ECE is the expected distance between the left and
right hand side of the calibration definition with respect to $d$.

Usually, the ECE is used to analyze classification models.[^GPSW17][^VWALRS19]
In this case, $P_X$ and $\mathcal{L}(Y \,|\, P_X)$ can be identified with vectors
in the probability simplex and $d$ can be chosen as a the cityblock distance,
the total variation distance, or the squared Euclidean distance.

For other probabilistic predictive models such as regression models, one has to
choose a more general distance measure $d$ between probability distributions on the
target space since the conditional distributions $\mathcal{L}(Y \,|\, P_X)$ can be
arbitrarily complex in general.

[^GPSW17]: Guo, C., et al. (2017). [On calibration of modern neural networks](http://proceedings.mlr.press/v70/guo17a.html). In *Proceedings of the 34th International Conference on Machine Learning* (pp. 1321-1330).

[^VWALRS19]: Vaicenavicius, J., et al. (2019). [Evaluating model calibration in classification](http://proceedings.mlr.press/v89/vaicenavicius19a.html). In *Proceedings of Machine Learning Research (AISTATS 2019)* (pp. 3459-3467).

[^WLZ21]: Widmann, D., Lindsten, F., & Zachariah, D. (2021). [Calibration tests beyond classification](https://openreview.net/forum?id=-bxf89v3Nx). To be presented at *ICLR 2021*.

## Estimators

The main challenge in the estimation of the ECE is the estimation of the conditional
distribution $\mathcal{L}(Y \,|\, P_X)$ from a finite data set of predictions and
corresponding targets. Typically, predictions are binned and empirical estimates of
the conditional distributions are calculated for each bin. You can construct such
estimators with [`ECE`](@ref).

```@docs
ECE
```

### Binning algorithms

Currently, two binning algorithms are supported. [`UniformBinning`](@ref) is a binning
schemes with bins of fixed bins of uniform size whereas [`MedianVarianceBinning`](@ref)
splits the validation data set of predictions and targets dynamically to reduce the
variance of the predictions.

```@docs
UniformBinning
MedianVarianceBinning
```
