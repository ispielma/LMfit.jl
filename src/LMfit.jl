"""
LMfit main module

In writing this I copied much about the python version, but using Symbols to label parameters internally.
"""
module LMfit

    using OrderedCollections
    using FillArrays

    using LsqFit
    using LsqFit: LsqFitResult
    using StatsAPI
    using StatsAPI: coef, confint, dof, nobs, rss, stderror, weights, residuals, vcov

    # In the future this will be split into several files, but for now I will develop in the main module

    export Constant, Parameter, Expression, Parameters, Model
    export add!, update_vars!, make_params, fit, eval_uncertainty
    export @generate_model

    export coef, confint, dof, nobs, rss, stderror, weights, residuals, vcov, mse, isconverged

    #
    # Define structures and their interfaces
    #

    # Definition of individal parameter class
    include("parameter_objects.jl")
    import .ParameterObjects: AbstractParameter, AbstractIndependentParameter, Parameter, ParameterWithUncertanty, Constant, Expression, IndependentVariable, Parameters
    import .ParameterObjects: validate, add!, find_dependencies!, resolve_parameters, _update_params_from_vect!

    # Definition of individal parameter class
    include("model_objects.jl")
    import .ModelObjects: Model, update_vars!, @generate_model, make_params, _strip_vars_kwargs

    #=
    .########.####.########....##.....##..#######..########..########.##......
    .##........##.....##.......###...###.##.....##.##.....##.##.......##......
    .##........##.....##.......####.####.##.....##.##.....##.##.......##......
    .######....##.....##.......##.###.##.##.....##.##.....##.######...##......
    .##........##.....##.......##.....##.##.....##.##.....##.##.......##......
    .##........##.....##.......##.....##.##.....##.##.....##.##.......##......
    .##.......####....##.......##.....##..#######..########..########.########
    =#

    """
        FitModel

    A wrapped Model that can be evaluated using the syntax expected by LsqFit, this is 
    an internal object that is used only by fit()
    """
    struct FitModel
        m::Model
        ps_fit::Parameters
        var_data::Dict{Symbol, Any} # Independent, i.e., x variables in the same order as keys are
        kwargs::Dict{Symbol, Any} # keyword arguments to passed into the function

        # Internal "private" fields
        _params_exposed_names::Vector{Symbol} # Parameters exposed to LsqFit
        _num_params_exposed::Int # Number of exposed free parameters note that because parameters can be vectors this is not always the number of exposed parameters
        _vect_to_params::Function # a function that takes the exposed parameters as a vector and returns the parameters in the right order
    end
    function FitModel(m::Model, ps::Parameters; kwargs=Dict(), key_args...) # kwargs are the x variables
        # Organize the independent variables
        (vars, not_vars) = _strip_vars_kwargs(m, key_args)

        # convert to a dict
        vars = Dict(vars)

        ps_fit = Parameters(ps) # create a copy so we don't change the original data
        validate(ps_fit)

        # Sort the parameters in order that they need to be determined
        find_dependencies!(ps_fit)

        # Get the names and number the parameters that are exposed to LsqFit
        params_exposed_names::Vector{Symbol} = [name for (name, p) in ps_fit if typeof(p) <: AbstractIndependentParameter]
        num_params_exposed = sum(length(ps_fit[name]) for name in params_exposed_names)

        # Get the mapping function
        vect_to_params = resolve_parameters(ps_fit)

        FitModel(m, ps_fit, vars, kwargs, params_exposed_names, num_params_exposed, vect_to_params)
    end

    """
        (f::FitModel)(x, p) = f(p; f.var_data...)

    Callable method that uses the format expected by LsqFit

    x is ignored, and is present only to match the call signature required by LsqFit
    p is a vector of the exposed parameters
    """
    (f::FitModel)(x, p) = f(p; f.var_data...)[:] # flattened as expected by LsqFit

    """
        f::FitModel)(p; kwargs...)

    Callable method that uses a format more similar to the remainder of this package where
    independent variables are passed as keyword arguments

    p is a vector of the exposed parameters as their native type, not Parameter objects.
    """
    function(f::FitModel)(p; kwargs...) # More canonical method where the independent variables are passed as keyword arguments
        if length(p) != f._num_params_exposed
            println(p)
            error("length of p=$(length(p)) must be the same length of the number of exposed parameters = $(f._num_params_exposed)")
        end

        # update parameters
        params = @invokelatest f._vect_to_params(p)

        _update_params_from_vect!(f.ps_fit, :value, params)

        # evaluate function
        return f.m(f.ps_fit; kwargs..., f.kwargs...)
    end

    """
        ModelResult

    returned by fit()
    """
    struct ModelResult{R,J,W,T <: AbstractArray}
        m::Model
        fm::FitModel
        ps_init::Parameters
        ps_best::Parameters

        resid::R
        jacobian::J
        covar_inv::T
        wt::W

        converged::Bool
        size::Tuple{Vararg{Int}}
        # size::NTuple{N,Int} where N 

        lfr::LsqFit.LsqFitResult

        function ModelResult(
            m::Model,
            fm::FitModel,
            ps_init::Parameters, 
            resid::R,
            jacobian::J,
            covar_inv::T, 
            wt::W, 
            converged::Bool,
            size::Tuple{Vararg{Int}}, 
            lfr::LsqFit.LsqFitResult) where {R,J,W,T <: AbstractArray}

            ps_best = Parameters()
            ps_fit = fm.ps_fit
            # set the order of the best parameters to match the initial, user provided order
            for k in keys(ps_init)
                p = ps_fit[k]
                if p isa AbstractIndependentParameter
                    # Inject uncertainties
                    add!(ps_best, ParameterWithUncertanty(p))
                else
                    add!(ps_best, p)
                end
            end

            # Update the uncertainties.  Notice that we are using the ordering provided by ps_fit
            ks_indep =  [k for (k, p) in ps_fit if p isa AbstractIndependentParameter] # these are the keys for variables that were actually minimized
            _update_params_from_vect!(ps_best, :σ, ks_indep, stderror(lfr) )

            new{R,J,W,T}(m, fm, ps_init, ps_best, resid, jacobian, covar_inv, wt, converged, size, lfr)
        end
    end
    ModelResult(fm::FitModel, ps_init, covar_inv, wt, size, lfr::LsqFit.LsqFitResult) = ModelResult(fm.m, fm, ps_init, reshape(lfr.resid, size...), lfr.jacobian, covar_inv, wt, lfr.converged, size, lfr)

    # Model result evaluates to the best fit
    # evaluate at the independent variables defined by kwargs...
    (mr::ModelResult)(; kwargs...) = mr.m(mr.ps_best; kwargs...)

    function Base.show(io::IO, mr::ModelResult)
        println(io, "ModelResult")
        println(io, "$(mr.ps_best)")
    end

    # Generally the statistics from LsqFit.jl is OK, but I need to update some of them.
    StatsAPI.coef(mr::ModelResult) = coef(mr.lfr)
    StatsAPI.rss(mr::ModelResult) = rss(mr.lfr)
    StatsAPI.weights(mr::ModelResult) = mr.wt # Note that this is not the uncertainties
    StatsAPI.residuals(mr::ModelResult) = mr.resid 
    StatsAPI.vcov(mr::ModelResult) = vcov(mr.lfr)
    StatsAPI.stderror(mr::ModelResult) = stderror(mr.lfr)
    StatsAPI.confint(mr::ModelResult; kwargs...) = confint(mr.lfr; kwargs...)
    isconverged(mr::ModelResult) = isconverged(mr.lfr)

    # use the Kish effective sample size
    StatsAPI.nobs(mr::ModelResult) = sum(mr.wt)^2 / sum(mr.wt.^2)
    StatsAPI.dof(mr::ModelResult) = nobs(mr) - length(coef(mr))
    mse(mr::ModelResult) = rss(mr) / nobs(mr) # TODO: This is not quite right!  Think of this as a weighted average
    chi2(mr::ModelResult) = rss(mr) / dof(mr) # TODO: Need to verify this.  Perhaps RSS is what should be fixed.
    χ2(mr::ModelResult) = chi2(mr::ModelResult)

    # StatsAPI.rss(lfr::LsqFitResult) = sum(abs2, lfr.resid) # lfr.resid includes weights AND uncertainties.
    # StatsAPI.residuals(lfr::LsqFitResult) = lfr.resid  # lfr.resid includes weights AND uncertainties.

    """
        eval_uncertainty(mr::ModelResult; kwargs...)
    
    Compute the the uncertainty for the fitted model (i.e. confidence bands)

    the independent variable are passed as keywords as usual for this package

    * n_σ : number of sigma's for the uncertainty band (currently ignored).

    * dscale : derivative step as a fraction of the uncertainty for each parameter

    """
    eval_uncertainty(mr::ModelResult; kwargs...) = eval_uncertainty(mr, 1.0; kwargs...)
    function eval_uncertainty(mr::ModelResult, n_σ; dscale=0.01, kwargs...)
        
        p = coef(mr) # vector of exposed parameters
        covar = vcov(mr) # covariance matrix of these parameters
        dp = stderror(mr) .* dscale # derivative step size
        f = mr.fm(p; kwargs...) # function value

        df = [copy(f) for i in eachindex(p)]

        # numerically compute derivatives
        for i in eachindex(p)
            p[i] += dp[i]
            df[i] .-= mr.fm(p; kwargs...)
            df[i] ./= dp[i]
            p[i] -= dp[i]
        end

        # Compute the uncertainty
        err = zero(f)
        for i in eachindex(p), j in eachindex(p)
            scale = covar[i, j] # need to scale by school function using σ, the number of sigmas
            err += (df[i] .* df[j]) .* scale
        end

        # Currently returns something using the function I need to differentiate with respect to the parameters
        return sqrt.(err) # for n_σ this multiplies the so-called school function.
    end

    """
        fit(m::Model, ydata, ps::Parameters; kwargs...)
        fit(m::Model, ydata, covar_inv, ps::Parameters; kwargs...)

    Fits the Model m to ydata, with the x values provided via kwargs, for example `x=x`` would 
    indicate that the name of the parameter in the function is actually `x`.

    covar_inv::AbstractArray the inverse of variance vector or covariance matrix.

    wt::AbstractArray (optional) : weights (such as a window) for each y point.

    scale_covar (bool, optional) : Assuming the provided weights are only proportionally correct.  When calculating uncertainties
        assume χ^2 was supposed to be 1, and scale the covariance matrix accordingly (default is false).
    """

    # Alias for old version where weight was an optional argument.
    fit(m::Model, ydata::AbstractArray, wt::AbstractArray, ps::Parameters; key_args...) = fit(m, ydata, ps; wt=wt, key_args...)
    
    function fit(m::Model, ydata::AbstractArray, ps_init::Parameters; kwargs=Dict(), covar_inv=nothing, wt=nothing, scale_covar=false, key_args...)

        # Organize the independent variables
        (vars, not_vars) = _strip_vars_kwargs(m, key_args)

        fm = FitModel(m, ps_init; vars..., kwargs)

        # Obtain vector of initial parameters and bounds
        p0 = vcat([p.value for (_, p) in fm.ps_fit if typeof(p) <: AbstractIndependentParameter]...)
        lb = vcat([p.min for (_, p) in fm.ps_fit if typeof(p) <: AbstractIndependentParameter]...)
        ub = vcat([p.max for (_, p) in fm.ps_fit if typeof(p) <: AbstractIndependentParameter]...)

        #
        # do curve fit
        #

        y_flat = ydata[:]

        # Make lazily-allocated arrays full of 1.0 with the same shape as ydata
        covar_inv = covar_inv === nothing ? Fill(1.0, size(ydata)) : covar_inv
        wt        = wt        === nothing ? Fill(1.0, size(ydata)) : wt

        # Flatten for LsqFit; `vec` keeps them lazy — still no data copied.
        covar_inv_flat = vec(covar_inv)   # Fill{Float64,1}
        wt_flat        = vec(wt)          # Fill{Float64,1}

        if wt===nothing
            result = curve_fit(fm, [], y_flat, p0; lower=lb, upper=ub, not_vars...)
        else
            result = curve_fit(fm, [], y_flat, covar_inv_flat .* wt_flat, p0; lower=lb, upper=ub, not_vars...)
        end

        params = @invokelatest fm._vect_to_params(result.param)
        
        _update_params_from_vect!(fm.ps_fit, :value, params)

        # TODO: implement scale_covar logic
        if scale_covar
            error("scale_covar not implemented yet")
        end

        ModelResult(fm, ps_init, covar_inv, wt, size(ydata), result)
    end

end # end of module
