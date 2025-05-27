# LMfit

I have recently being curve fitting in Julia using the very nice [LsqFit.jl](https://github.com/JuliaNLSolvers/LsqFit.jl) package.  The main issue is that it requires a lot of boiler plate coding to take a general function into a form that can be accepted.

Inspired by the python [lmfit](https://lmfit.github.io/lmfit-py/intro.html) package, this Julia package aims to provide an intuitive wrapper around `LsqFit.jl`. 

An example notebook is given here [Demo.ipynb](notebooks/Demo.ipynb)

## Uncertainties 

Unfortunately fitting packages use weights and uncertainties in a nearly interchangeable way which is almost always invalid.  Before moving forward and understanding how this package will resolve these issues, it is important to identify what the issues are, and why they don't matter for a great many people.

The basic idea is that we are minimizing the weighted error function
$$
\begin{align}
E &= \sum_{ij}[y_i - m(x_i; \Lambda)] \Sigma^{-1}_{ij}[y_j - m(x_j; \Lambda)] \rightarrow \sum_j \left[\frac{y_j - m(x_j; \Lambda)}{\sigma_j}\right]^2,
\end{align}
$$
where $y_j$ is the vector of $N$ measurements sampled at points $x_j$ with covariance matrix $\Sigma^{-1}_{ij}$, and $m(x_j; \lambda)$ is the model function which depends some set of optimization parameters $\Lambda$. 
The second expression is what we usually usually see in terms of point-point uncertainties.

With this we can define the standard quantities:

* Number of observations: ${N_\mathrm{obs}} = N$; so far a trivial statement; becomes non-trivial below.
* Residuals: $\Delta_j \equiv y_j - m(x_j; \Lambda)$.
* Weighted residuals: $\bar \Delta_i \equiv \sum_j \Sigma^{-1/2}_{ij} \Delta_j$, giving the loss function $E = \sum_i \bar \Delta_i^2$.  Because the covariance matrix is symmetric one can compute its square root via the Cholesky transform.  For the case of uncorrelated uncertainties this really is just the usual weighted residual.
* Residual sum of squares (unweighted): $\mathrm{RSS} \equiv \sum_j \Delta_j^2$; notice that this is unweighted.
* Mean squared error (unweighted): $\mathrm{MSE} \equiv\mathrm{RSS}  /  N.
* Degrees of freedom: $\mathrm{DoF} = {N_\mathrm{obs}} - \mathrm{Dim}(\Lambda)$, i.e., the difference between the number of observations and the number of model parameters.
* Reduced chi-squared: $\chi^2 \equiv \mathrm{DoF}^{-1} \sum_i \bar \Delta_i^2 = E / \mathrm{DoF}$. i.e., the loss function weighted by DoF.

In this context most fitting packages accept a `weights` argument of some sort, equal to the inverse covariance matrix $\Sigma^{-1}_{ij}$.  For many applications, this is great, but already it is useful to notice that the `Lsqfit.jl` package does not conform to these standard definitions.  Given a fit result `lfr` of type `LsqFitResult` we instead have:
* `lfr.resid` = weighted residuals $\bar \Delta$.
* `rss(lfr)` = chi-squared numerator $\sum_j \bar \Delta_j^2$.
* `mse(lfr)` = $(\sum_j \bar \Delta_j^2) / \mathrm{DoF}$, really the unbiased residual-variance estimator for the weighted problem.

Here, the `LMfit.jl` package reverses the non-standard conventions.

 Even aside these issues, having a single `weights` variable becomes problematic when one wants to apply a window function $w_j$.  The reason for this is that a window function is designed to reject (and potentially re-weight) data for reasons other than the expected statistical uncertainty.  This has the important consequence of in effect reducing number of independent observations to the Kish effective sample size
 $$
 \begin{align}
 N_{\rm eff} = \frac{(\sum_j w_j)^2}{\sum_j w_j^2}
 \end{align}
 $$
 and changes $\mathrm{DoF} =  N_{\rm eff} - \mathrm{Dim}(\Lambda)$