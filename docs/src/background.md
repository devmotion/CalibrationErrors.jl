# Probabilistic setting

## Probabilistic model

A probabilistic model predicts a probability distribution of possible outputs
for a given input.

A very simple probabilistic model is a model that predicts a uniform 
distribution for a dice roll; there is no input and the possible outputs are
the numbers $1,2,3,4,5,6$. A probably more complicated probabilistic model
would be a model that predicts the distribution of stock price changes from
the stock prices of last week; here the input are the stock prices of last week
and the possible outputs are formed by the set of possible stock price changes.

If we assume that for each input there exists an underlying
unknown probability distribution that captures the relation between the given
input and the possible outputs, we can aim for a model whose predicted
distributions are, in some sense, close to these unknown probability distributions.

More mathematically, if we view inputs and outputs as random variables $X$ and $Y$
on some probability space, we can strive for a model $g$ that satisfies
```math
    g(X) \approx \mu_{Y|X}(\cdot|X),
```
where $\mu_{Y|X}$ is a version of the conditional distribution of $Y$ given $X$.

Closeness of $g(X)$ and $\mu_{Y|X}(\cdot|X)$ can be measured with distance measures
of probability distributions. Two prominent families of distances are
[$\phi$-divergences](https://en.wikipedia.org/wiki/F-divergence)
and [integral probability metrics](https://arxiv.org/pdf/0901.2698.pdf).

## Classification model

Here we restrict ourselves to classification models, i.e., models for which output
$Y$ takes only values from a finite set.

The dice roll model above is a classification model, whereas the model that predicts
stock price changes is not.