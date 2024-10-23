"""
LMfit main module

In writing this I copied much about the python version, but using Symbols to label parameters internally.
"""
module LMfit

using LsqFit
using LsqFit: LsqFitResult
using StatsAPI
using StatsAPI: coef, confint, dof, nobs, rss, stderror, weights, residuals, vcov

# In the future this will be split into several files, but for now I will develop in the main module

export Constant, Parameter, Expression, Parameters, Model, FitModel
export add!, update_vars!, make_params, fit
export @generate_model

export coef, confint, dof, nobs, rss, stderror, weights, residuals, vcov, mse, isconverged

#
# Define structures and their interfaces
#

# Definition of individal parameter class
include("parameter_objects.jl")
import .ParameterObjects: Parameter, Constant, Expression, IndependentVariable, Parameters
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

Returned by fit and contains the result of the fit as well as a wrapped Model that can be evaluated using the symtax expected by LsqFit
"""
struct FitModel
    m::Model
    ps_fit::Parameters
    var_data::Dict{Symbol, Any} # Independent, i.e., x variables in the same order as keys are

    # Internal "private" fields
    _params_exposed_names::Vector{Symbol} # Parameters exposed to LsqFit
    _num_params_exposed::Int # Number of exposed free paramteres note that because parameters can be vectors this is not always the number of exposed parameters
    _vect_to_params::Function # a function that takes the exposed parameters as a vector and returns the parameters in the right order
end
function FitModel(m::Model, ps::Parameters; kwargs...) # kwargs are the x variables
    # Organize the independent variables
    (vars, not_vars) = _strip_vars_kwargs(m, kwargs)

    # convert to a dict
    vars = Dict(vars)

    ps_fit = Parameters(ps) # create a copy so we don't change the origional data
    validate(ps_fit)

    # Sort the parameters in order that they need to be determined
    find_dependencies!(ps_fit)

    # Get the names and number the parameters that are exposed to LsqFit
    params_exposed_names::Vector{Symbol} = [name for (name, p) in ps_fit if typeof(p) <: Parameter]
    num_params_exposed = sum(length(ps_fit[name]) for name in params_exposed_names)

    # Get the mapping function
    vect_to_params = resolve_parameters(ps_fit)

    FitModel(m, ps_fit, vars, params_exposed_names, num_params_exposed, vect_to_params)
end

function(f::FitModel)(x, p) # x is there only because it is required for the curve_fit function
    if length(p) != f._num_params_exposed
        println(p)
        error("length of p=$(length(p)) must be the same length of the number of exposed parameters = $(f._num_params_exposed)")
    end

    # update parameters
    params = @invokelatest f._vect_to_params(p)

    _update_params_from_vect!(f.ps_fit, params)

    # evaluate function and flatten the output
    return f.m(f.ps_fit; f.var_data...)[:]
end

"""
    ModelResult

"""
struct ModelResult
    m::Model
    ps_init::Parameters
    ps_best::Parameters
    size::Tuple{Vararg{Int}}
    lfr::LsqFit.LsqFitResult
end

function Base.show(io::IO, mr::ModelResult)
    println(io, "ModelResult")
    println(io, "$(mr.ps_best)")

end

StatsAPI.coef(mr::ModelResult) = coef(mr.lfr)
StatsAPI.nobs(mr::ModelResult) = nobs(mr.lfr)
StatsAPI.dof(mr::ModelResult) = dof(mr.lfr)
StatsAPI.rss(mr::ModelResult) = rss(mr.lfr)
StatsAPI.weights(mr::ModelResult) = weights(mr.lfr)
StatsAPI.residuals(mr::ModelResult) = reshape(residuals(mr.lfr), mr.size...)
StatsAPI.vcov(mr::ModelResult) = vcov(mr.lfr)
StatsAPI.stderror(mr::ModelResult) = stderror(mr.lfr)
StatsAPI.confint(mr::ModelResult; kwargs...) = confint(mr.lfr; kwargs...)
mse(mr::ModelResult) = mse(mr.lfr)
isconverged(mr::ModelResult) = isconverged(mr.lfr)


"""
    fit(m::Model, y, ps::Parameters; kwargs...)

Fits the Model m to the data y, with the provided x values.  Note that the LMFit.py syntax
is smarter in that the kw argument actually define the name of the paramteres that will be used for the x values 
via the name of keyword arguments: 

so x=x would indicate that the name of the parameter in the function is actually x.
"""
fit(m::Model, ydata::AbstractArray, ps::Parameters; kwargs...) = fit(m::Model, ydata, nothing, ps::Parameters; kwargs...) # TODO: need a nicer way to the weights

function fit(m::Model, ydata::AbstractArray, wt, ps::Parameters; kwargs...)

    # Organize the independent variables
    (vars, not_vars) = _strip_vars_kwargs(m, kwargs)

    fm = FitModel(m, ps; vars...)

    # Obtain vector of initial parameters and bounds
    p0 = vcat([p.value for (_, p) in fm.ps_fit if typeof(p) <: Parameter]...)
    lb = vcat([p.min for (_, p) in fm.ps_fit if typeof(p) <: Parameter]...)
    ub = vcat([p.max for (_, p) in fm.ps_fit if typeof(p) <: Parameter]...)

    # do curve fit
    if wt===nothing
        result = curve_fit(fm, [], ydata[:], p0; lower=lb, upper=ub, not_vars...)
    else
        result = curve_fit(fm, [], ydata[:], wt[:], p0; lower=lb, upper=ub, not_vars...)
    end

    params = @invokelatest fm._vect_to_params(result.param)
    
    println(params)

    _update_params_from_vect!(fm.ps_fit, params)

    ModelResult(fm.m, ps, fm.ps_fit, size(ydata), result)
end

end # end of module
