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
calibration of the largest predictions $\max_y g_y(x)$ of a model $g$ is considered.
According to this common notion a model is calibrated if almost always
```math
    \mathbb{P}[Y = \textrm{arg} \, \max_y g_y(X) \,|\, \max_y g_y(X)] = \max_y g_y(X).
```

### Strong notion

According to the more general definition by
[Bröcker (2009)](https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1002/qj.456)
and [Vaicenavicius et al.](http://proceedings.mlr.press/v89/vaicenavicius19a/vaicenavicius19a.pdf),
a probabilistic model $g$ is calibrated if almost always
```math
    \mathbb{P}[Y = y \,|\, g(X)] = g_y(X)
```
for all classes $y$.