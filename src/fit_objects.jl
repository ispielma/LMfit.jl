"""
    FitObjects

Fit preparation, the LsqFit adapter, fitted results, result statistics, and
confidence-band evaluation.
"""
module FitObjects

    import Distributions: TDist, quantile
    import SpecialFunctions: erf

    using FillArrays: Fill
    using LsqFit: curve_fit
    import LsqFit
    using StatsAPI
    using StatsAPI: coef, confint, dof, nobs, residuals, rss, stderror, vcov, weights

    import ..ParameterObjects: AbstractIndependentParameter, Parameters
    import ..ParameterObjects: ParameterWithUncertanty, _update_params_from_vect!
    import ..ParameterObjects: add!, find_dependencies!, resolve_parameters, validate
    import ..ModelObjects: Model, _route_fit_kwargs, _strip_vars_kwargs

    export FitModel, ModelResult, eval_uncertainty, fit, isconverged, mse

    """
        FitModel

    Adapt a `Model` and its named parameters to LsqFit's flat-vector calling
    convention. This is an internal object created by `fit`.
    """
    struct FitModel
        m::Model
        ps_fit::Parameters
        var_data::Dict{Symbol, Any}
        kwargs::Dict{Symbol, Any}

        _params_exposed_names::Vector{Symbol}
        _num_params_exposed::Int
        _vect_to_params::Function
    end

    function FitModel(m::Model, ps::Parameters; kwargs=Dict(), key_args...)
        vars, _ = _strip_vars_kwargs(m, key_args)
        var_data = Dict(vars)

        ps_fit = Parameters(ps)
        validate(ps_fit)
        find_dependencies!(ps_fit)

        names = Symbol[]
        for (name, p) in ps_fit
            if p isa AbstractIndependentParameter
                push!(names, name)
            end
        end

        num_params = 0
        for name in names
            num_params += length(ps_fit[name])
        end

        resolver = resolve_parameters(ps_fit)
        return FitModel(m, ps_fit, var_data, kwargs, names, num_params, resolver)
    end

    (f::FitModel)(x, p) = f(p; f.var_data...)[:]

    function (f::FitModel)(p; kwargs...)
        if length(p) != f._num_params_exposed
            println(p)
            error(
                "length of p=$(length(p)) must be the same length of " *
                "the number of exposed parameters = $(f._num_params_exposed)",
            )
        end

        params = f._vect_to_params(p)
        _update_params_from_vect!(f.ps_fit, :value, params)
        return f.m(f.ps_fit; kwargs..., f.kwargs...)
    end

    """Result returned by `fit`."""
    struct ModelResult{R, J, W, T <: AbstractArray}
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
            lfr::LsqFit.LsqFitResult,
        ) where {R, J, W, T <: AbstractArray}
            ps_best = Parameters()
            ps_fit = fm.ps_fit
            for name in keys(ps_init)
                p = ps_fit[name]
                if p isa AbstractIndependentParameter
                    add!(ps_best, ParameterWithUncertanty(p))
                else
                    add!(ps_best, p)
                end
            end

            names = Symbol[]
            for (name, p) in ps_fit
                if p isa AbstractIndependentParameter
                    push!(names, name)
                end
            end
            _update_params_from_vect!(ps_best, :σ, names, stderror(lfr))

            return new{R, J, W, T}(
                m,
                fm,
                ps_init,
                ps_best,
                resid,
                jacobian,
                covar_inv,
                wt,
                converged,
                size,
                lfr,
            )
        end
    end

    function ModelResult(fm::FitModel, ps_init, covar_inv, wt, size, lfr)
        resid = reshape(lfr.resid, size...)
        return ModelResult(
            fm.m,
            fm,
            ps_init,
            resid,
            lfr.jacobian,
            covar_inv,
            wt,
            lfr.converged,
            size,
            lfr,
        )
    end

    (mr::ModelResult)(; kwargs...) = mr.m(mr.ps_best; kwargs...)

    function Base.show(io::IO, mr::ModelResult)
        println(io, "ModelResult")
        println(io, "$(mr.ps_best)")
    end

    StatsAPI.coef(mr::ModelResult) = coef(mr.lfr)
    StatsAPI.rss(mr::ModelResult) = rss(mr.lfr)
    StatsAPI.weights(mr::ModelResult) = mr.wt
    StatsAPI.residuals(mr::ModelResult) = mr.resid
    StatsAPI.vcov(mr::ModelResult) = vcov(mr.lfr)
    StatsAPI.stderror(mr::ModelResult) = stderror(mr.lfr)
    StatsAPI.confint(mr::ModelResult; kwargs...) = confint(mr.lfr; kwargs...)
    isconverged(mr::ModelResult) = isconverged(mr.lfr)

    StatsAPI.nobs(mr::ModelResult) = sum(mr.wt)^2 / sum(mr.wt .^ 2)
    StatsAPI.dof(mr::ModelResult) = nobs(mr) - length(coef(mr))
    mse(mr::ModelResult) = rss(mr) / nobs(mr)
    chi2(mr::ModelResult) = rss(mr) / dof(mr)
    χ2(mr::ModelResult) = chi2(mr)

    """
        eval_uncertainty(mr::ModelResult; kwargs...)
        eval_uncertainty(mr::ModelResult, n_σ; dscale=0.01, kwargs...)

    Evaluate the confidence-band width for the fitted model. Independent
    variables are supplied with the same keywords used to evaluate the model.
    """
    eval_uncertainty(mr::ModelResult; kwargs...) =
        eval_uncertainty(mr, 1.0; kwargs...)

    function eval_uncertainty(mr::ModelResult, n_σ; dscale=0.01, kwargs...)
        p = copy(coef(mr))
        covar = vcov(mr)
        dp = stderror(mr) .* dscale
        f = collect(eltype(covar), mr.fm(p; kwargs...))

        df = Vector{typeof(f)}(undef, length(p))
        try
            for i in eachindex(p)
                p_step = copy(p)
                p_step[i] += dp[i]
                df[i] = (f .- mr.fm(p_step; kwargs...)) ./ dp[i]
            end
        finally
            mr.fm(p; kwargs...)
        end

        err = zero(f)
        for i in eachindex(p), j in eachindex(p)
            scale = covar[i, j]
            err += (df[i] .* df[j]) .* scale
        end

        prob = erf(n_σ / sqrt(2))
        scale = quantile(TDist(dof(mr)), (1 + prob) / 2)
        return sqrt.(err) .* scale
    end

    """
        fit(m::Model, ydata, ps::Parameters; kwargs...)
        fit(m::Model, ydata, wt, ps::Parameters; kwargs...)

    Fit `m` to `ydata`. Independent variables use their declared names.
    Declared model keywords may also be passed directly; the legacy
    `kwargs=Dict(...)` form remains supported. The positional array overload is
    retained as the existing window-weight interface.
    """
    fit(m::Model, ydata::AbstractArray, wt::AbstractArray, ps::Parameters; key_args...) =
        fit(m, ydata, ps; wt, key_args...)

    function fit(
        m::Model,
        ydata::AbstractArray,
        ps_init::Parameters;
        kwargs=Dict(),
        covar_inv=nothing,
        wt=nothing,
        scale_covar=false,
        key_args...,
    )
        vars, context, fit_opts = _route_fit_kwargs(m, key_args)
        model_context = Dict{Symbol, Any}()
        for (name, value) in kwargs
            model_context[name] = value
        end
        for (name, value) in context
            model_context[name] = value
        end
        fm = FitModel(m, ps_init; vars..., kwargs=model_context)

        p0 = _free_field(fm.ps_fit, :value)
        lb = _free_field(fm.ps_fit, :min)
        ub = _free_field(fm.ps_fit, :max)

        y_flat = ydata[:]
        covar_inv = covar_inv === nothing ? Fill(1.0, size(ydata)) : covar_inv
        wt = wt === nothing ? Fill(1.0, size(ydata)) : wt

        covar_inv_flat = vec(covar_inv)
        wt_flat = vec(wt)
        result = curve_fit(
            fm,
            [],
            y_flat,
            covar_inv_flat .* wt_flat,
            p0;
            lower=lb,
            upper=ub,
            fit_opts...,
        )

        if scale_covar
            error("scale_covar not implemented yet")
        end

        params = fm._vect_to_params(result.param)
        _update_params_from_vect!(fm.ps_fit, :value, params)
        return ModelResult(fm, ps_init, covar_inv, wt, size(ydata), result)
    end

    function _free_field(ps::Parameters, field::Symbol)
        result = Float64[]
        for p in values(ps)
            if p isa AbstractIndependentParameter
                value = getfield(p, field)
                if value isa Number
                    push!(result, value)
                else
                    append!(result, value)
                end
            end
        end
        return result
    end

end
