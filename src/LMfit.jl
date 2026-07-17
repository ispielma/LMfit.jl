"""
LMfit main module

In writing this I copied much about the python version, but using Symbols to label parameters internally.
"""
module LMfit

    export Constant, Parameter, Expression, Parameters, Model
    export add!, update_vars!, make_params, fit, eval_uncertainty
    export @generate_model

    export coef, confint, dof, nobs, rss, stderror, weights, residuals, vcov
    export mse, isconverged

    using StatsAPI: coef, confint, dof, nobs, residuals, rss, stderror, vcov, weights

    include("parameter_objects.jl")
    import .ParameterObjects: AbstractIndependentParameter, AbstractParameter
    import .ParameterObjects: Constant, Expression, IndependentVariable, Parameter
    import .ParameterObjects: Parameters, ParameterWithUncertanty, add!
    import .ParameterObjects: _update_params_from_vect!, find_dependencies!
    import .ParameterObjects: resolve_parameters, validate

    include("model_objects.jl")
    import .ModelObjects: Model, @generate_model, make_params, update_vars!
    import .ModelObjects: _route_fit_kwargs, _strip_vars_kwargs

    include("fit_objects.jl")
    import .FitObjects: FitModel, ModelResult, chi2, eval_uncertainty, fit
    import .FitObjects: isconverged, mse, χ2

end
