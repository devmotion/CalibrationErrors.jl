# # Classification of penguin species
#
# ## Packages

using AlgebraOfGraphics
using CairoMakie
using CalibrationErrors
using DataFrames
using Distributions
using PalmerPenguins
using Query

using Random

## Set color cycle globally
set_theme!(; palette=(color=Makie.wong_colors(0.8)[1:3],))
CairoMakie.activate!(; type="svg")

# ## Data
#
# In this example we study the calibration of different models that classify three penguin
# species based on measurements of their bill and flipper lengths.
#
# We use the [Palmer penguins dataset](https://allisonhorst.github.io/palmerpenguins/) to
# to train and validate the models.

penguins = dropmissing(DataFrame(PalmerPenguins.load()))

plt =
    data(penguins) *
    mapping(
        :bill_length_mm => "bill length (mm)", :flipper_length_mm => "flipper length (mm)"
    ) *
    mapping(; color=:species) *
    visual(; alpha=0.7)
draw(plt)

# We split the data randomly into a training and validation dataset. The training dataset
# contains around 60% of the samples.

Random.seed!(1234)
n = nrow(penguins)
k = floor(Int, 0.7 * n)
Random.seed!(100)
train = shuffle!(vcat(trues(k), falses(n - k)))

penguins.train = train

train_penguins = penguins[train, :]
val_penguins = penguins[.!train, :];

#-

dataset = :train => renamer(true => "training", false => "validation") => "Dataset"
plt =
    data(df) *
    mapping(
        :bill_length_mm => "bill length (mm)", :flipper_length_mm => "flipper length (mm)"
    ) *
    mapping(; color=:species, col=dataset) *
    visual(; alpha=0.7)
draw(plt)

# ## Fitting normal distributions
#
# For each species, we fit independent normal distributions to the observations of the bill
# and flipper length in the training data, using maximum likelihood estimation.

penguins_fit = @from i in train_penguins begin
    @group i by i.species into g
    @select {
        species = key(g),
        proportion = length(g) / nrow(train_penguins),
        bill = fit(Normal, g.bill_length_mm),
        flipper = fit(Normal, g.flipper_length_mm),
    }
    @collect DataFrame
end

# We plot the estimated normal distributions.

function xrange(dists, alpha=0.0001)
    xmin = minimum(Base.Fix2(quantile, alpha), dists)
    xmax = maximum(Base.Fix2(quantile, 1 - alpha), dists)
    return range(xmin, xmax; length=1_000)
end

function plot_normal_fit(dists, species, xlabel)
    f = Figure()
    Axis(f[1, 1]; xlabel=xlabel, ylabel="density")
    xs = xrange(dists)
    plots = map(dists) do dist
        ys = pdf.(dist, xs)
        l = lines!(xs, ys)
        b = band!(xs, 0, ys)
        return [l, b]
    end
    Legend(f[1, 2], plots, species, "species")
    return f
end

plot_normal_fit(penguins_fit.bill, penguins_fit.species, "bill length [mm]")

#-

plot_normal_fit(penguins_fit.flipper, penguins_fit.species, "flipper length [mm]")

# ## Naive Bayes classifier
#
# Let us assume that the bill and flipper length are conditionally independent given the
# penguin species. Then Bayes' theorem implies that
# ```math
# \begin{aligned}
# \mathbb{P}(\mathrm{species} \,|\, \mathrm{bill}, \mathrm{flipper})
# &= \frac{\mathbb{P}(\mathrm{species}) \mathbb{P}(\mathrm{bill}, \mathrm{flipper} \,|\, \mathrm{species})}{\mathbb{P}(\mathrm{bill}, \mathrm{flipper})} \\
# &= \frac{\mathbb{P}(\mathrm{species}) \mathbb{P}(\mathrm{bill} \,|\, \mathrm{species}) \mathbb{P}(\mathrm{flipper} \,|\, \mathrm{species})}{\mathbb{P}(\mathrm{bill}, \mathrm{flipper})}.
# \end{aligned}
# ```
# This predictive model is known as
# [naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier).
#
# In the section above, we estimated $\mathbb{P}(\mathrm{species})$,
# $\mathbb{P}(\mathrm{bill} \,|\, \mathrm{species})$, and
# $\mathbb{P}(\mathrm{flipper} \,|\, \mathrm{species})$ for each penguin species from
# the training data. For the conditional distributions we used a Gaussian approximation.

function predict_naive_bayes_classifier(fit, data)
    ## Compute unnormalized probabilities
    z =
        log.(permutedims(fit.proportion)) .+
        logpdf.(permutedims(fit.bill), data.bill_length_mm) .+
        logpdf.(permutedims(fit.flipper), data.flipper_length_mm)

    ## Normalize probabilities
    u = maximum(z; dims=2)
    z .= exp.(z .- u)
    sum!(u, z)
    z ./= u

    return DataFrame(z, fit.species)
end

train_predict = predict_naive_bayes_classifier(penguins_fit, train_penguins)
val_predict = predict_naive_bayes_classifier(penguins_fit, val_penguins);

# ## Evaluation
#
# We evaluate the probabilistic predictions of the naive Bayes classifier that we just
# trained. It is easier to work with a numerical encoding of the true penguin species and a
# corresponding vector of predictions.

train_species = convert(Vector{Int}, indexin(train_penguins.species, names(train_predict)))
train_probs = RowVecs(Matrix{Float64}(train_predict))

val_species = convert(Vector{Int}, indexin(val_penguins.species, names(val_predict)))
val_probs = RowVecs(Matrix{Float64}(val_predict));

# ### Log-likelihood
#
# We compute the average log-likelihood of the training and validation data. It is
# equivalent to the negative cross-entropy.

function mean_loglikelihood(species, probs)
    return mean(log(p[s]) for (s, p) in zip(species, probs))
end

mean_loglikelihood(train_species, train_probs)

#-

mean_loglikelihood(val_species, val_probs)

# ### Brier score
#
# The average log-likelihood is also equivalent to the
# [logarithmic score](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).
# The Brier score is another strictly proper scoring rule that can be used for evaluating
# probabilistic predictions.

function brier_score(species, probs)
    return mean(
        sum(abs2(pi - (i == s)) for (i, pi) in enumerate(p)) for
        (s, p) in zip(species, probs)
    )
end

brier_score(train_species, train_probs)

#-

brier_score(val_species, val_probs)

# ### Expected calibration error
#
# As all proper scoring rules, the logarithmic and the Brier score can be [decomposed in
# three terms that quantify the sharpness and calibration of the predictive model and the
# irreducible uncertainty of the targets that is inherent to the prediction
# problem](https://doi.org/10.1002/qj.456). The calibration term in this decomposition is
# the expected calibration error (ECE)
# ```math
# \mathbb{E} d\big(P_X, \mathrm{law}(Y \,|\, P_X)\big)
# ```
# with respect to the score divergence $d$.
#
# Scoring rules, however, include also the sharpness and the uncertainty term. Thus models
# can trade off calibration for sharpness and therefore scoring rules are not suitable for
# specifically evaluating calibration of predictive models.
#
# The score divergence to the logarithmic and the Brier score are the Kullback-Leibler (KL)
# divergence
# ```math
# d\big(P_X, \mathrm{law}(Y \,|\, P_X)\big) = \sum_{y} \mathbb{P}(Y = y \,|\, P_X)
# \log\big(\mathbb{P}(Y = y \,|\, P_X) / P_X(\{y\})\big)
# ```
# and the squared Euclidean distance
# ```math
# d\big(P_X, \mathrm{law}(Y \,|\, P_X)\big) = \sum_{y} \big(P_X - \mathrm{law}(Y \,|\, P_X)\big)^2(\{y\}),
# ```
# respectively. The KL divergence is defined only if $\mathrm{law}(Y \,|\, P_X)$ is
# absolutely continuous with respect to $P_X$, i.e., if $P_X(\{y\}) = 0$ implies
# $\mathbb{P}(Y = y \,|\, P_X) = 0$.

# We estimate the ECE by binning the probability simplex of predictions $P_X$ and computing
# the weighted average of the distances between the mean prediction and the distribution of
# targets in each bin.
#
# One approach is to use bins of uniform size.

ece = ECE(UniformBinning(10), (μ, y) -> kl_divergence(y, μ))
ece(train_probs, train_species)

#-

ece(val_probs, val_species)

# For the squared Euclidean distance we obtain:

ece = ECE(UniformBinning(10), SqEuclidean())
ece(train_probs, train_species)

#-

ece(val_probs, val_species)

# Alternatively, one can use a data-dependent binning scheme that tries to split the
# predictions in a way that minimizes the variance in each bin.
#
# With the KL divergence we get:

ece = ECE(MedianVarianceBinning(5), (μ, y) -> kl_divergence(y, μ))
ece(train_probs, train_species)

#-

ece(val_probs, val_species)

# For the squared Euclidean distance we obtain:

ece = ECE(MedianVarianceBinning(5), SqEuclidean())
ece(train_probs, train_species)

#-

ece(val_probs, val_species)

# We see that the estimates (of the same theoretical quantity!) are highly dependent on the
# chosen binning scheme.

# ### Kernel calibration error
#
# As an alternative to the ECE, we estimate the kernel calibration error (KCE). We keep it
# simple here, and use the tensor product kernel
# ```math
# k\big((\mu, y), (\mu', y')\big) = \delta_{y,y'} \exp{\bigg(-\frac{{\|\mu - \mu'\|}_2^2}{2\nu^2} \bigg)}
# ```
# with length scale $\nu > 0$ for predictions $\mu,\mu'$ and corresponding targets $y, y'$.
# For simplicity, we estimate length scale $\nu$ with the median heuristic.

distances = pairwise(SqEuclidean(), train_probs)
λ = sqrt(median(distances[i] for i in CartesianIndices(distances) if i[1] < i[2]))
kernel = (GaussianKernel() ∘ ScaleTransform(inv(λ))) ⊗ WhiteKernel();

# We obtain the following biased estimates of the squared KCE (SKCE):

skce = BiasedSKCE(kernel)
skce(train_probs, train_species)

#-

skce(val_probs, val_species)

# Similar to the biased estimates of the ECE, the biased estimates of the SKCE are always
# non-negative. The unbiased estimates can be negative as well, in particular if the model
# is (close to being) calibrated:

skce = UnbiasedSKCE(kernel)
skce(train_probs, train_species)

#-

skce(val_probs, val_species)

# When the datasets are large, the quadratic sample complexity of the standard biased and
# unbiased estimators of the SKCE can become prohibitive. In these cases, one can resort to
# an estimator that averages estimates of non-overlapping blocks of samples. This estimator
# allows to trade off computational cost for increased variance.
#
# Here we consider the extreme case of blocks with two samples, which yields an estimator
# with linear sample complexity:

skce = BlockUnbiasedSKCE(kernel, 2)
skce(train_probs, train_species)

#-

skce(val_probs, val_species)
