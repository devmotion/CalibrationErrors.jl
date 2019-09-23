# CalibrationErrors.jl

Estimation of calibration errors.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://devmotion.github.io/CalibrationErrors.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://devmotion.github.io/CalibrationErrors.jl/dev)
[![Build Status](https://travis-ci.com/devmotion/CalibrationErrors.jl.svg?branch=master)](https://travis-ci.com/devmotion/CalibrationErrors.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/devmotion/CalibrationErrors.jl?svg=true)](https://ci.appveyor.com/project/devmotion/CalibrationErrors-jl)
[![DOI](https://zenodo.org/badge/188981243.svg)](https://zenodo.org/badge/latestdoi/188981243)
[![Codecov](https://codecov.io/gh/devmotion/CalibrationErrors.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/devmotion/CalibrationErrors.jl)
[![Coveralls](https://coveralls.io/repos/github/devmotion/CalibrationErrors.jl/badge.svg?branch=master)](https://coveralls.io/github/devmotion/CalibrationErrors.jl?branch=master)

## Overview

This package implements different estimators of the expected calibration error
(ECE) and the squared kernel calibration error calibration error (SKCE) in the
Julia language.

## Example

Calibration errors can be estimated from a data set of predicted probabilities
and and a set of labels by executing
```julia
calibrationerror(estimator, data)
```

The data set `data` has to be a tuple of predictions and labels. The predictions
have to be provided as a matrix of size `(m, n)`, in which each of the `n`
columns corresponds to predicted probabilities of the labels `1,…,m`. The
corresponding labels have to be provided as a vector of length `n`, in which
every element is from the set `1,…,m`.

This package implements the estimator `ECE` of the ECE and the estimators
`BiasedSKCE`, `QuadraticUnbiasedSKCE`, and `LinearUnbiasedSKCE` for the SKCE.
