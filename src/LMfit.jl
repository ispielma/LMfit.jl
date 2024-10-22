"""
LMfit main module

In writing this I copied much about the python version, but using Symbols to label parameters internally.
"""
module LMfit

using LsqFit

# In the future this will be split into several files, but for now I will develop in the main module

export Constant, Parameter, Expression, Parameters, Model, FitModel
export add!, update_vars!, make_params, fit
export @generate_model

#
# Define structures and their interfaces
#

# Definition of individal parameter class
include("parameter_objects.jl")
import .ParameterObjects: Parameter, Constant, Expression, IndependentVariable, Parameters
import .ParameterObjects: validate, add!, find_dependencies!, resolve_parameters

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

a wrapped Model that can be evaluated using the symtax expected by LsqFit
"""
struct FitModel
    params_exposed_names::Vector{Symbol} # Parameters exposed to LsqFit
    m::Model
    ps::Parameters
    var_data::Dict{Symbol, Any} # Independent, i.e., x variables in the same order as keys are
    vect_to_params::Function # a function that takes the exposed parameters as a vector and returns the parameters in the right order
end
function FitModel(m::Model, ps::Parameters; kwargs...) # kwargs are the x variables
    # Organize the independent variables
    (vars, not_vars) = _strip_vars_kwargs(m, kwargs)

    # convert to a dict
    vars = Dict(vars)

    # Sort the parameters in order that they need to be determined
    find_dependencies!(ps)

    # Get the names of the parameters that are exposed to LsqFit
    params_exposed_names::Vector{Symbol} = [name for (name, p) in ps if typeof(p) <: Parameter]

    ps_new = deepcopy(ps)
    validate(ps_new)

    # Get the mapping function
    vect_to_params = resolve_parameters(ps_new)

    FitModel(params_exposed_names, m, ps_new, vars, vect_to_params)
end

function(f::FitModel)(x, p) # x is there only because it is required for the curve_fit function
    if length(p) != length(f.params_exposed_names)
        error("length of p must be the same length of the number of exposed parameters")
    end

    # update parameters
    params = @invokelatest f.vect_to_params(p)

    for (p, value) in zip(values(f.ps), params)
        p.value = value
    end

    # evaluate function and flatten the output
    return f.m(f.ps; f.var_data...)[:]
end


"""
    fit(m::Model, y, ps::Parameters; kwargs...)

Fits the Model m to the data y, with parameters ps with the provided x values.  Note that the LMFit.py syntax
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
    p0 = [p.value for (_, p) in fm.ps if typeof(p) <: Parameter]
    lb = [p.min for (_, p) in fm.ps if typeof(p) <: Parameter]
    ub = [p.max for (_, p) in fm.ps if typeof(p) <: Parameter]
   
    # do curve fit
    if wt===nothing
        result = curve_fit(fm, [], ydata[:], p0; lower=lb, upper=ub, not_vars...)
    else
        result = curve_fit(fm, [], ydata[:], wt[:], p0; lower=lb, upper=ub, not_vars...)
    end

    for (j, name) in enumerate(fm.params_exposed_names)
        fm.ps[name].value = result.param[j]
    end
    fm.ps
end


end # end of module
