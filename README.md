# CalibrationErrors.jl

Estimation of calibration errors.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://devmotion.github.io/CalibrationErrors.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://devmotion.github.io/CalibrationErrors.jl/dev)
[![Build Status](https://travis-ci.com/devmotion/CalibrationErrors.jl.svg?branch=master)](https://travis-ci.com/devmotion/CalibrationErrors.jl)
[![Build Status](https://github.com/devmotion/CalibrationErrors.jl/workflows/CI/badge.svg)](https://github.com/devmotion/CalibrationErrors.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![DOI](https://zenodo.org/badge/188981243.svg)](https://zenodo.org/badge/latestdoi/188981243)
[![Codecov](https://codecov.io/gh/devmotion/CalibrationErrors.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/devmotion/CalibrationErrors.jl)
[![Coveralls](https://coveralls.io/repos/github/devmotion/CalibrationErrors.jl/badge.svg?branch=master)](https://coveralls.io/github/devmotion/CalibrationErrors.jl?branch=master)
[![Bors enabled](https://bors.tech/images/badge_small.svg)](https://app.bors.tech/repositories/24611)

## Overview

This package implements different estimators of the expected calibration error
(ECE) and the squared kernel calibration error calibration error (SKCE) in the
Julia language.

## Example

Calibration errors can be estimated from a data set of predicted probabilities
and and a set of targets by executing
```julia
calibrationerror(estimator, predictions, targets)
```

The predictions can be provided as a vector of `n` vectors of predicted probabilities
of length `m` or as a matrix of size `(m, n)`, in which each of the `n` columns corresponds
to predicted probabilities of the targets `1,…,m`. The corresponding targets have to be
provided as a vector of length `n`, in which every element is from the set `1,…,m`.

Alternatively, it is possible to specify a tuple of predictions and targets or a vector of
tuples of predictions and targets.

This package implements the estimator `ECE` of the ECE and the estimators
`BiasedSKCE`, `UnbiasedSKCE`, and `BlockUnbiasedSKCE` for the SKCE.
