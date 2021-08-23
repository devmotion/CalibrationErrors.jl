# CalibrationErrors.jl

Estimation of calibration errors.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://devmotion.github.io/CalibrationErrors.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://devmotion.github.io/CalibrationErrors.jl/dev)
[![Build Status](https://github.com/devmotion/CalibrationErrors.jl/workflows/CI/badge.svg?branch=main)](https://github.com/devmotion/CalibrationErrors.jl/actions?query=workflow%3ACI+branch%3Amain)
[![DOI](https://zenodo.org/badge/188981243.svg)](https://zenodo.org/badge/latestdoi/188981243)
[![Codecov](https://codecov.io/gh/devmotion/CalibrationErrors.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/devmotion/CalibrationErrors.jl)
[![Coveralls](https://coveralls.io/repos/github/devmotion/CalibrationErrors.jl/badge.svg?branch=main)](https://coveralls.io/github/devmotion/CalibrationErrors.jl?branch=main)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![Bors enabled](https://bors.tech/images/badge_small.svg)](https://app.bors.tech/repositories/24611)

**There are also [Python](https://github.com/devmotion/pycalibration) and [R](https://github.com/devmotion/rcalibration) interfaces for this package**

## Overview

This package implements different estimators of the expected calibration error
(ECE), the squared kernel calibration error (SKCE), and the
unnormalized calibration mean embedding (UCME) in the Julia language.

## Example

Calibration errors can be estimated from a data set of predicted probability distributions
and a set of corresponding observed targets by executing
```julia
estimator(predictions, targets)
```

The sets of predictions and targets have to be provided as vectors.

This package implements the estimator `ECE` of the ECE, the estimators
`BiasedSKCE`, `UnbiasedSKCE`, and `BlockUnbiasedSKCE` for the SKCE, and `UCME` for the
UCME.

## Related packages

[CalibrationErrorsDistributions.jl](https://github.com/devmotion/CalibrationErrorsDistributions.jl)
extends calibration error estimation in this package to more general probabilistic
predictive models that output arbitrary probability distributions.

[CalibrationTests.jl](https://github.com/devmotion/CalibrationTests.jl) implements
statistical hypothesis tests of calibration.

[pycalibration](https://github.com/devmotion/pycalibration) is a Python interface for CalibrationErrors.jl, CalibrationErrorsDistributions.jl, and CalibrationTests.jl.

[rcalibration](https://github.com/devmotion/rcalibration) is an R interface for CalibrationErrors.jl, CalibrationErrorsDistributions.jl, and CalibrationTests.jl.

## Talk at JuliaCon 2021

<center>
<iframe width="560" style="height:315px" src="https://www.youtube-nocookie.com/embed/PrLsXFvwzuA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</center>

The slides of the talk are available as [Pluto notebook](https://talks.widmann.dev/2021/07/calibration/).

## References

If you use CalibrationsErrors.jl as part of your research, teaching, or other activities,
please consider citing the following publications:

Widmann, D., Lindsten, F., & Zachariah, D. (2019). [Calibration tests in multi-class
classification: A unifying framework](https://proceedings.neurips.cc/paper/2019/hash/1c336b8080f82bcc2cd2499b4c57261d-Abstract.html). In
*Advances in Neural Information Processing Systems 32 (NeurIPS 2019)* (pp. 12257–12267).

Widmann, D., Lindsten, F., & Zachariah, D. (2021).
[Calibration tests beyond classification](https://openreview.net/forum?id=-bxf89v3Nx).
*International Conference on Learning Representations (ICLR 2021)*.
