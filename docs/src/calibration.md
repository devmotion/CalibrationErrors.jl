# Calibration

## Motivation

Ideally one would like to have a model that predicts the underlying probability
distribution for almost every input, i.e., a model $g$ such that almost always
```math
    g(X) = \mu_{Y|X}(\cdot|X).
```
Unfortunately, if we try to infer the model from a finite data set of inputs and
outputs that is usually not possible.

In safety-critical applications such as medical decision-making or autonomous
driving, however, important decisions are based on the predictions of a model.
Since we are not able to obtain the perfect model, the model has to satisfy
other properties such that it is deemed trustworthy.

## Definition

One such property is calibration, which is also called reliability. Loosely speaking,
if we repeatedly predict the same distribution for different pairs of inputs and
outputs, we would like that in the long run the empirical distribution of the
observed outputs is similar to the predicted probability distribution. This property
guarantees that the predicted distribution is not only an arbitrary probability
distribution but actually makes sense from a frequentist point of view.

A [classic example from the literature](https://www.jstor.org/stable/2987588) is a
weather forecaster who each morning predicts the probability that it will rain during
the day. If we assume that the forecaster's predictions are observed for a long time,
the forecaster is called calibrated "if among those days for which his prediction
is $x$, the long-run relative frequency of rain is also $x$".

### Common notion

Commonly (see, e.g,
[Guo et al. (2017)](http://proceedings.mlr.press/v70/guo17a/guo17a.pdf)), only
calibration of the most-confident predictions $\max_y g_y(x)$ of a model $g$ is
considered. According to this common notion a model is calibrated if almost
always
```math
    \mathbb{P}[Y = \textrm{arg} \, \max_y g_y(X) \,|\, \max_y g_y(X)] = \max_y g_y(X).
```

### Strong notion

According to the more general definition by
[Bröcker (2009)](https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1002/qj.456)
and [Vaicenavicius et al. (2019)](http://proceedings.mlr.press/v89/vaicenavicius19a/vaicenavicius19a.pdf),
a probabilistic model $g$ is calibrated if almost always
```math
    \mathbb{P}[Y = y \,|\, g(X)] = g_y(X)
```
for all classes $y$.

For classification problems with more than two classes, this definition of
calibration is stronger than the more common one above. By reducing the model
and applying the strong notion to the simplified model, however, this definition
still allows to investigate the calibration of the model with respect to only
certain aspects of interest such as the calibration of the most-confident
predictions.

Thus in this Julia package and its documentation, we always refer to the strong
notion of calibration.

Let $y_1, \ldots, y_m$ be the possible outputs. Then we can also define
calibration in a vectorized form. Equivalently to the definition above, a model
$g$ is calibrated if and only if
```math
    r(g(X)) - g(X) = 0
```
holds almost always, where
```math
    r(\xi) := (\mathbb{P}[Y = y_1 \,|\, g(X) = \xi], \ldots, \mathbb{P}[Y = y_m \,|\, g(X) = \xi])
```
denotes the so-called calibration function.

## Measures

Calibration measures allow a more fine-tuned analysis of calibration and enable
comparisons of calibration of different models. Intuitively, calibration
measures quantify the deviation of the left and right hand side in the
definitions above.

### Expected calibration error (ECE)

The most common calibration measure is the so-called expected calibration error
(ECE) (see, e.g.,
[Guo et al. (2017)](http://proceedings.mlr.press/v70/guo17a/guo17a.pdf)).
Informally, it is defined as the average distance between the left and right
hand side of the definition above with respect to some metric. Mathematically,
the expected calibration of model $g$ with respect to distance measure $d$ is
defined as
```math
    \mathrm{ECE}[d, g] := \mathbb{E}[d(r(g(X)), g(X))].
```
Here $d$ could be, e.g., the cityblock distance, the total variation distance,
or the squared Euclidean distance.

If $d(p, q) = 0$ if and only if $p = q$, then the ECE of model $g$ with respect
to distance measure $d$ is zero if and only if $g$ is calibrated.

### Calibration error (CE)

More generally, Widmann et al. (2019) define the calibration error (CE) of
a model $g$ with respect to a function class $\mathcal{F} \subset \{f \colon
\Delta^m \to \mathbb{R}^m\}$ as
```math
    \mathrm{CE}[\mathcal{F}, g] := \sup_{f \in \mathcal{F}} \mathbb{E}[(r(g(X)) - g(X))^\intercal f(g(X))].
```

If model $g$ is calibrated, then the CE is zero, regardless of the choice of
$\mathcal{F}$. However, for some function spaces (e.g., for
$\mathcal{F} = \{0\}$) the CE is zero even if $g$ is not calibrated.

Interestingly, the ECE with respect to the cityblock distance, the total
variation distance, and the squared Euclidean distance are all special cases
of the CE (Widmann et al. (2019)).

### Kernel calibration error (KCE)

The kernel calibration error (KCE) is another special case of the CE, in which
the unit ball of a reproducing kernel Hilbert space (RKHS) of vector-valued
functions is chosen as function space $\mathcal{F}$.

A RKHS of vector-valued functions $f \colon \Delta^m \to \mathbb{R}^m$ can be
identified with a unique matrix-valued kernel $k \colon \Delta^m \times
\Delta^m \to \mathbb{R}^{m \times m}$. Then the KCE of a model $g$ with respect
to kernel $k$ is defined as
```math
    \mathrm{KCE}[k, g] := \mathrm{CE}[\mathcal{F}, g],
```
where $\mathcal{F}$ is the unit ball of the RKHS corresponding to kernel $k$.

As Widmann et al. (2019) show, for a large class of kernels (so-called universal
kernels) the KCE is zero if and only if the model $g$ is calibrated. Moreover,
the KCE can be formulated in terms of the kernel $k$ as
```math
    \mathrm{KCE}[k, g] := {\left(\mathbb{E}[(e_Y - g(X))^{\intercal} k(g(X), g(X')) (e_{Y'} - g(X'))]\right)}^{1/2},
```
where $(X',Y')$ is an independent copy of $(X,Y)$ and $e_i$ denotes the $i$th
unit vector.

The so-called maximum mean calibration error (MMCE), proposed by
[Kumar et al. (2018)](http://proceedings.mlr.press/v80/kumar18a/kumar18a.pdf),
can be viewed as a special case of the KCE, in which only the most-confident
predictions are considered (Widmann et al. (2019)).
