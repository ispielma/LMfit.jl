# LMfit MVP Status

Last reviewed: 2026-07-17

LMfit's modeling core already supports the main demonstrated use cases: named independent variables, scalar and vector parameters, bounds, constants, expression-derived parameters, multidimensional model outputs, and confidence bands. The remaining MVP work is concentrated in statistical semantics, interface clarity, validation, and test coverage.

## Resolved decisions and fixes

### Default fits use unit inverse variance

Omitting `covar_inv` intentionally means that every observation has inverse variance `1`. This is the desired definition of an unweighted fit in LMfit. The unreachable `wt === nothing` branch has been removed, and a regression test verifies that the default fit matches an explicit unit inverse-variance fit.

This decision also means that LMfit does not ask LsqFit to infer an unknown residual variance for default fits. Parameter uncertainties retain the known-unit-variance interpretation.

### Confidence-band evaluation is side-effect-free

`eval_uncertainty` now perturbs copies of the fitted coefficient vector instead of mutating `coef(result)`. It no longer mutates model outputs, so models returning ranges are supported. The internal fit-model parameter state is restored in a `finally` block after successful or failed derivative evaluation.

Regression tests cover range-valued evaluation, finite output, coefficient preservation, and coefficient preservation after a perturbed model evaluation throws.

### Covariance scaling reports its actual status

`scale_covar=true` now raises the intended `ErrorException("scale_covar not implemented yet")` instead of a `FieldError`. Implementing covariance scaling remains open MVP work if this option is to remain public.

### Test harness is operational

The stale `LMfit1` import is fixed. Test-only dependencies are isolated in `test/Project.toml`, where LsqFit is pinned to the exact version used by the numerical golden tests; the package itself retains normal compatible-version bounds. `Pkg.test()` now runs successfully. The current tests are focused regressions rather than comprehensive coverage.

### Real-world calling patterns are protected

Compact synthetic tests now protect the distinct LMfit usage patterns found in the DSF example notebooks and Paper_MOT_Dataset without reproducing either analysis. Coverage includes default and positional-weight fits, fixed model context, zero and multiple independent variables, manual and macro-generated models, scalar and vector parameters, constants, expression-derived parameters with dotted exponentials and powers, multidimensional outputs, fitted-model evaluation, result statistics, and confidence bands.

The seeded demo tests separately lock the exact numerical coefficients for all five notebook fits with LsqFit 0.15.1.

### Architecture deepening is implemented

The production source continues to follow the one-module-per-Julia-file rule:

- `ParameterObjects` now owns dependency discovery, validation, flattening, reconstruction, and in-process expression evaluation. It no longer generates Julia source, calls `Meta.parse`, evaluates code globally, or requires `@invokelatest`.
- `ModelObjects` now has one invocation path and an authoritative fit-keyword router. Independent variables, declared model context, and optimizer options are separated in one place. Existing nested `kwargs=Dict(...)` calls remain supported.
- `FitObjects` is a self-contained module owning fit preparation, the LsqFit adapter, fitted-result reconstruction, result statistics, and confidence-band evaluation.
- `LMfit` is now the small package facade that includes the three complete child modules and preserves the existing exported interface and internal type bindings.

## Open findings

### P1: Separate inverse covariance from window weights

The positional overload in `src/LMfit.jl` records its array argument as `wt`. Existing consumers use that same position for different concepts:

- DSF notebooks pass inverse variances such as `errors.^-2`.
- Paper_MOT_Dataset passes pixel weights that combine statistical weighting with a selection window.

The optimizer receives the product `covar_inv .* wt`, so fitted coefficients can be correct. However, `nobs` applies the Kish effective-sample-size formula to `wt`. When an inverse-variance array arrives through the positional overload, varying measurement precision incorrectly changes `nobs`, `dof`, `mse`, and the Student-t scale used for confidence bands.

MVP completion requires one documented meaning for the positional overload or its deprecation in favor of explicit `covar_inv=` and `wt=` keywords. Tests should cover nonuniform inverse variances, window weights with zeros, and their combination.

### P1: Define and implement the statistics contract

`README.md` defines residuals as `y - model`, RSS as the unweighted residual sum of squares, MSE as `RSS / N`, and reduced chi-squared as the weighted loss divided by degrees of freedom. The current `ModelResult` methods instead expose LsqFit's weighted `model - y` residuals and weighted RSS. Its `mse` is weighted RSS divided by effective `nobs`, which is neither the documented MSE nor reduced chi-squared.

Before locking behavior with tests, decide the public definitions of:

- raw residuals and weighted residuals;
- RSS and weighted loss;
- MSE and reduced chi-squared;
- `weights(result)` when covariance and window weights are both present;
- `nobs` and `dof` when a window reduces the effective sample size.

MVP completion requires README and implementation agreement plus analytic tests against weighted linear regression.

### P1: Support or reject full covariance matrices explicitly

The `fit` docstring accepts an inverse covariance vector or matrix, but the implementation always flattens `covar_inv`. A full `N × N` matrix becomes length `N²` and fails when combined with `N` observations.

MVP completion requires either preserving matrix covariance for LsqFit's matrix-weight path or narrowing the documented interface to diagonal inverse-variance arrays with a clear validation error. Tests should exercise the selected contract.

### P1: Complete public-behavior test coverage

The demonstrated and real-world fitting paths are now protected. Remaining priority tests are:

1. Parameter validation, bounds, copying, ordering, and dependency failures.
2. Supported parameter conversions and explicit rejection of invalid conversions.
3. Missing, extra, and conflicting model/optimizer keyword errors.
4. Analytic result-statistics tests after the statistics contract is settled.

Tests should use public interfaces and avoid pinning internal helper names or call order.

### P2: Decide whether to implement `scale_covar`

The keyword is public and documented but intentionally throws. Either implement its stated reduced-chi-squared rescaling behavior with tests or remove it until supported. The current explicit error is safe but not an MVP feature.

### P2: Repair `isconverged`

`isconverged(result)` is exported but delegates to an LsqFit result method that does not exist in the pinned LsqFit version, so the call raises a `MethodError`. `ModelResult` already stores the convergence flag. Decide whether the public function should return that field or be removed, then add a public-behavior test.

### P2: Repair or remove invalid parameter conversions

Several conversion methods in `src/parameter_objects.jl` access fields that their source type does not have, including `min` and `max` on `Expression` or `Constant`. `Parameter(name, arg)` only prints a diagnostic and returns `nothing`.

MVP completion requires tests for supported conversions and deletion or correction of methods that cannot satisfy their declared interface.

### P3: Finish user-facing documentation

`docs/index.md` is empty, and the README ends during the discussion of statistical semantics. Documentation should describe the settled fit signature, covariance/window distinction, parameter expressions, result statistics, confidence bands, and compact examples derived from `notebooks/Demo.ipynb`.

## Architecture direction

The planned architecture deepening is complete and preserves the repository's one-module-per-Julia-file rule. `ParameterObjects`, `ModelObjects`, and `FitObjects` each own a cohesive namespace in one complete source file, while `LMfit` is the package facade. All changes are in-process; LsqFit remains the only external fitting adapter, and no additional abstraction layer or runtime dependency was introduced.
