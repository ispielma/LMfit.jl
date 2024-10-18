"""
LMfit main module

In writing this I copied much about the python version, but using Symbols to label parameters internally.
"""
module LMfit


using LsqFit
using ExprTools
using OrderedCollections

# In the future this will be split into several files, but for now I will develop in the main module

export Parameter, Parameters, Model, FitModel
export add!, update_vars!, find_dependencies!, make_params, fit
export @generate_model

#
# Define structures and their interfaces
#

# Definition of individal parameter class
include("parameter_objects.jl")
using .ParameterObjects: Parameter, AbstractParameter

#=
.########.....###....########.....###....##.....##.########.########.########.########...######.
.##.....##...##.##...##.....##...##.##...###...###.##..........##....##.......##.....##.##....##
.##.....##..##...##..##.....##..##...##..####.####.##..........##....##.......##.....##.##......
.########..##.....##.########..##.....##.##.###.##.######......##....######...########...######.
.##........#########.##...##...#########.##.....##.##..........##....##.......##...##.........##
.##........##.....##.##....##..##.....##.##.....##.##..........##....##.......##....##..##....##
.##........##.....##.##.....##.##.....##.##.....##.########....##....########.##.....##..######.
=#

struct Parameters
    parameters::OrderedDict{Symbol, AbstractParameter}
end
Parameters() = Parameters(OrderedDict{Symbol, Parameter}())
Parameters(args...; kwargs...) = add!(Parameters(), args...; kwargs...)

function add!(ps::Parameters, p::AbstractParameter)
    ps.parameters[p.name] = p
    ps
end
function add!(ps::Parameters, name::Symbol; kwargs...)
    ps.parameters[name] = Parameter(name; kwargs...)
    ps
end
function add!(ps::Parameters, pvec::Vector{AbstractParameter})
    for p in pvec
        add!(ps, p)
    end
    ps
end

"""
    validate

check for error conditions
"""
function validate(ps::Parameters)
    for (name, p) in ps
        if name != p.name
            error("item $(name): Parameters name-key must match name-field the associated record")
        end

        if p.min >= p.max
            error("item $(name): p.min must be less than p.max")
        end

        if p.value >= p.max || p.value <= p.min
            error("item $(name): p.value must be between p.min and p.max")
        end

        if p.vary==true && p.expr != :()
            error("Cannot vary and specify an expression")
        end
    end
end

function _get_symbols(ex)
    list = []
    walk!(list) = ex -> begin
       ex isa Symbol && push!(list, ex)
       ex isa Expr && ex.head == :call && map(walk!(list), ex.args[2:end])
       list
    end
    Set{Symbol}(walk!([])(ex))
end

"""
    find_dependencies!(ps::Parameters)

This is a key part of the logic of this package.  It will iterate over the parameters to see if they are
fully determined, and to check for error conditions such as circular dependencies.

It will then resort the parameters in the order that they need to be resolved if there are no errors
"""
function find_dependencies!(ps::Parameters)

    # Find dependencies
    for p in values(ps.parameters)
        if p.vary
            p._depends_on = Set{Symbol}()
        else
            p._depends_on = _get_symbols(p.expr)
        end
    end

    unsorted_parameters = copy(ps.parameters) # not a deep copy
    sorted_parameters = empty(unsorted_parameters) # create an empty version of unsorted_parameters

    resolved_one = true
    while resolved_one 
        resolved_one = false

        # find resolved dependencies
        for (name, p) in unsorted_parameters
            if isempty(p._depends_on)
                sorted_parameters[name] = p
                delete!(unsorted_parameters, name)
                resolved_one = true
            end
        end

        # remove resolved variables from depends_on sets
        for unsorted in values(unsorted_parameters)
            for sorted in values(sorted_parameters)
                delete!(unsorted._depends_on, sorted.name)
            end
        end
    end

    if !isempty(unsorted_parameters)
        error("Circular dependencies detected")
    end

    empty!(ps.parameters)
    for (name, p) in sorted_parameters
        ps.parameters[name] = p
    end

    return ps
end

"""
Creates a function that evaluates a vector of varied parameters and returns a vector of all the parameters
"""
function resolve_parameters(ps::Parameters)
    
    inputs = [p for p in values(ps) if p.vary]
    constants = [p for p in values(ps) if !p.vary && p.expr == :()]
    expressions = [p for p in values(ps) if p.expr != :()]

     # we take a vector of parameters
    prog = "(params) -> begin\n"

    # we unpack the adjustable parameters into their associated variables
    lines = ["   $(p.name) = params[$(i)]\n" for (i, p) in enumerate(inputs)]
    prog *= join(lines)
    prog *= "\n"

    # we unpack the constant parameters into their associated variables
    lines = ["   $(p.name) = $(p.value)\n" for (i, p) in enumerate(constants)]
    prog *= join(lines)
    prog *= "\n"

    # we unpack the expression parameters into their associated variables
    lines = ["   $(p.name) = $(string(p.expr))\n" for (i, p) in enumerate(expressions)]
    prog *= join(lines)
    prog *= "\n"

    # we pack these up into a single array
    prog *= "   result = Vector{Float64}(undef, $(length(ps)))\n"
    lines = ["   result[$(i)] = $(p.name)\n" for (i, p) in enumerate(values(ps))]
    prog *= join(lines)
    prog *= "\n"
    
    prog *= "   return result\n" 
    prog *= "end"

    body = Meta.parse(prog)

    eval(body)
end


#
# New methods for existing Base functions
#

function Base.getindex(ps::Parameters, indices...)
    ps.parameters[indices...]
end
Base.iterate(ps::Parameters) = iterate(ps.parameters)
Base.iterate(ps::Parameters, state) = iterate(ps.parameters, state)
Base.keys(ps::Parameters) = keys(ps.parameters)
Base.length(ps::Parameters) = length(ps.parameters)
Base.values(ps::Parameters) = values(ps.parameters)

function Base.show(io::IO, ps::Parameters)
    println(io, "Parameters:")
    for p in values(ps.parameters)
        println(io, "\t$(String(p))")
    end
end

#=
.##.....##..#######..########..########.##......
.###...###.##.....##.##.....##.##.......##......
.####.####.##.....##.##.....##.##.......##......
.##.###.##.##.....##.##.....##.######...##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##..#######..########..########.########
=#

"""
Model

Wraps a function of some paramters into something of the form that a fitter will accept

Owing to the multiple dispatch nature of Julia, argument names cannot be identified without a function call.

var_names are the names of the independent variables
"""
mutable struct Model
    name::String
    func::Function
    arg_names::Vector{Symbol} # args in order
    kwarg_names::Vector{Symbol} # kwargs in order
    var_names::OrderedDict{Symbol, Int} # Independent variables (in order), pointing to their location in arg_names
    param_names::OrderedDict{Symbol, Int} # Parameters (in order), pointing to their location in arg_names
end
function Model(name::String, func::Function, arg_names::Vector{Symbol}, kwarg_names::Vector{Symbol}; varnames=[])

    if !allunique(kwarg_names)
        error("kwarg_names must be unique")
    end

    if !allunique(arg_names)
        error("arg_names must be unique")
    end

    param_names = OrderedDict(k => v for (v, k) in enumerate(arg_names))

    m = Model(name, func, arg_names, kwarg_names, OrderedDict{Symbol, Int}(), param_names)

    # to reuse logic we now update the var_names
    update_vars!(m, varnames...)
end
"""
Allow the model to be evaluated given a set of parameters, or just a vector of numerical values for the parameters

x is a vector/tuple/whatever of independent variables
"""
(m::Model)(ps::Parameters; kwargs...) = m([ps[name].value for (name, _) in m.param_names];  kwargs...) # generate list of paramteres, ordered as expected

"""
p is a vector of parameter _values_
kwargs contains the independent (i.e., x, y, ... variables) with keys equal to their names
"""
function (m::Model)(p::Vector; kwargs...)

    if !(m.var_names ⊆ keys(kwargs))
        error("model.arg_names = $(m.arg_names) must be a subset of kwargs = $(keys(kwargs))")
    end

    # Pull out independent variables from kwargs
    vars = Dict(name => kwargs[name] for name in keys(m.var_names))

    if length(p) != length(m.param_names)
        error("p must be same length as Model.param_names")
    end

    args = Vector{Any}(undef, length(m.arg_names))
    for ( (_, index), param) in zip(m.param_names, p)
        args[index] = param
    end

    # Get independent variables
    for (name, index) in m.var_names
        args[index] = pop!(kwargs, name)
    end

    m.func(args...; kwargs...)
end

"""
update_vars!(m::Model, var_names)

Update the independent variables of the model
"""
function update_vars!(m::Model, var_names...)

    if !allunique(var_names)
        error("var_names must be unique")
    end

    if !(var_names ⊆ m.arg_names)
        error("var_names=$(var_names) must be a subset of model.arg_names = $(m.arg_names)")
    end

    var_indices = indexin(var_names, m.arg_names)
    m.var_names = OrderedDict(k => v for (k, v) in zip(var_names, var_indices) )

    # Make a dict of what is left over
    m.param_names = OrderedDict(k => v for (v, k) in enumerate(m.arg_names) if !haskey(m.var_names, k) )

    m
end




"""
    @generate_model func

This macro grabs the information needed for an lmfit model from a function definition.  I need a macro to
do the required introspection in Julia.
"""
macro generate_model(defun)
    def = splitdef(defun)
    name = def[:name]
    args = get(def, :args, [])
    kwargs = get(def, :kwargs, [])

    # because I there are two types of results possible as in args = Any[:(x::Float64), :amp, :cen, :wid]
    arg_names::Vector{Symbol} = [isa(arg, Expr) ? arg.args[1] : arg for arg in args]

    # because kwargs looks like Any[:($(Expr(:kw, :offset, 0.0))), :($(Expr(:kw, :cat, 2.0)))])
    kwarg_names::Vector{Symbol} = [isa(arg, Expr) ? arg.args[1] : arg for arg in kwargs]

    # Surround the original body with @info messages
    wrapped = quote
        $defun
        Model($(String(name)), $name, $arg_names, $kwarg_names)
    end

    # Recombine the function definition and output it (escaped!)
    esc(wrapped)
end

"""
make_params(m::Model)

make parameters for the model

kwargs are default values
"""
function make_params(m::Model; kwargs...)
    ps = Parameters()

    for param in keys(m.param_names)
        value = haskey(kwargs, param) ? kwargs[param] : -Inf
        add!(ps, param; value=value)
    end

    return ps
end

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

    # TODO: These lines are copied from above, so can be factored out to avoid duplication.
    if !(m.var_names ⊆ keys(kwargs))
        error("model.arg_names = $(m.arg_names) must be a subset of kwargs = $(keys(kwargs))")
    end

    # Pull out independent variables from kwargs
    var_data = Dict(name => kwargs[name] for name in keys(m.var_names))

    # Sort the parameters in order that they need to be determined
    find_dependencies!(ps)

    # Get the names of the parameters that are exposed to LsqFit
    params_exposed_names::Vector{Symbol} = [name for (name, p) in ps if p.vary]

    ps_new = deepcopy(ps)
    validate(ps_new)

    # Get the mapping function
    vect_to_params = resolve_parameters(ps_new)

    FitModel(params_exposed_names, m, ps_new, var_data, vect_to_params)
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
function fit(m::Model, ydata, ps::Parameters; kwargs...)

    fm = FitModel(m, ps; kwargs...)

    return fm

    # Obtain vector of initial parameters and bounds
    p0 = [p.value for (_, p) in fm.ps if p.vary]
    lb = [p.min for (_, p) in fm.ps if p.vary]
    ub = [p.max for (_, p) in fm.ps if p.vary]
   
    # do curve fit
    result = curve_fit(fm, [], ydata[:], p0, lower=lb, upper=ub)

    for (j, name) in enumerate(fm.params_exposed_names)
        fm.ps[name].value = result.param[j]
    end
    fm.ps
end

end # end of module
