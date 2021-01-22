# Other calibration errors

## Unnormalized calibration mean embedding (UCME)

Instead of the formulation of the calibration error as an integral
probability metric one can consider the unnormalized calibration
mean embedding (UCME).

Let $\mathcal{P} \times \mathcal{Y}$ be the product space of
predictions and targets. The UCME for a real-valued kernel
$k \colon (\mathcal{P} \times \mathcal{Y}) \times (\mathcal{P} \times \mathcal{Y}) \to \mathbb{R}$
and $m$ test locations is defined[^WLZ] as
```math
\mathrm{UCME}_{k,m}^2 := m^{-1} \sum_{i=1}^m \Big(\mathbb{E}_{Y,P_X} k\big(T_i, (P_X, Y)\big) - \mathbb{E}_{Z_X,P_X} k\big(T_i, (P_X, Z_X)\big)\Big)^2,
```
where test locations $T_1, \ldots, T_m$ are i.i.d. random variables whose
law is absolutely continuous with respect to the Lebesgue measure on
$\mathcal{P} \times \mathcal{Y}$.

The plug-in estimator of $\mathrm{UCME}_{k,m}^2$ is available as [`UCME`](@ref).

```@docs
UCME
```

[^WLZ]: Widmann, D., Lindsten, F., & Zachariah, D. (2021). [Calibration tests beyond classification](https://openreview.net/forum?id=-bxf89v3Nx). To be presented at *ICLR 2021*.
