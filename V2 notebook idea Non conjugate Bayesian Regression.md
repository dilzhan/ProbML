## V2 notebook idea: Non-conjugate Bayesian regression

Build a notebook demonstrating what happens when we move beyond conjugate Gaussian Bayesian linear regression.

### Main experiment

Compare regression models with the same Gaussian parameter prior

[
w \sim \mathcal N(0,\tau^2 I)
]

but different observation models:

* Gaussian likelihood → standard Bayesian mean regression, conjugate posterior.
* Laplace likelihood → robust/median regression, non-conjugate posterior.
* Optionally Student-(t) likelihood → robust heavy-tailed regression.

Generate linear data with several strong outliers and compare how the models behave.

For Laplace regression:

[
y_i\mid x_i,w \sim \operatorname{Laplace}(x_i^T w,b).
]

The Laplace likelihood corresponds to minimizing absolute deviations, so (x^Tw) represents the conditional median rather than the conditional mean.

### Inference

The Gaussian likelihood + Gaussian prior has an analytic Gaussian posterior.

Changing to a Laplace/Student-(t) likelihood breaks conjugacy:

[
p(w\mid D)\propto p(D\mid w)p(w)
]

is no longer Gaussian and its normalization/predictive integrals are generally intractable.

Use this as a practical application of the inference methods implemented earlier in the project:

* MH/MALA/HMC or another MCMC method to sample from the posterior.
* Laplace approximation where appropriate: approximate the posterior locally around the MAP as
  [
  p(w\mid D)\approx\mathcal N(w_{\text{MAP}},-H^{-1}).
  ]
  Student-(t) or logistic regression may be cleaner demonstrations of Laplace approximation because the Laplace likelihood is not twice differentiable at zero residuals.

Then perform posterior predictive sampling:

[
w^{(s)}\sim p(w\mid D),
\qquad
y_*^{(s)}\sim p(y_*\mid x_*,w^{(s)}).
]

Important distinction: MCMC/Laplace/etc. are **inference methods for obtaining/approximating the parameter posterior**. Posterior predictive sampling happens afterward using that inferred posterior.

### Possible extensions

Add asymmetric Laplace likelihood for Bayesian quantile regression, e.g. (\tau=0.1,0.5,0.9).

Other later experiments in the same spirit:

* Bayesian logistic regression: Bernoulli likelihood + Gaussian prior.
* Poisson regression.
* Gaussian likelihood + Laplace prior → Bayesian LASSO/sparsity.
* Heteroscedastic regression.
* Hierarchical/shrinkage priors.

### Point of the notebook

The conceptual story should be:

[
\text{change modeling assumptions}
\rightarrow
\text{break conjugacy}
\rightarrow
\text{non-Gaussian posterior}
\rightarrow
\text{approximate inference}
\rightarrow
\text{posterior predictive}.
]

The notebook should connect the earlier sampling and approximation notebooks to an actual modeling problem, rather than implementing inference algorithms only on toy target distributions.

