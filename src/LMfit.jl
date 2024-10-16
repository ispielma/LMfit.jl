"""
LMfit main module

In writing this I copied much about the python version, including using strings to label parameters internally.  I think
in Julia I should use symbols.
"""
module LMfit


using LsqFit
using ExprTools
using OrderedCollections

# In the future this will be split into several files, but for now I will develop in the main module

export Parameter, Parameters, Model
export add!, update_vars!, make_params
export @generate_model

#
# Define structures and their interfaces
#

"""
Parameter

This is the elemental component of the `LMfit` package.
"""
mutable struct Parameter
    name::String
    value::Float64
    vary::Bool
    min::Float64
    max::Float64
    expr::String
    user_data::Any
end
function Parameter(name::String; value=NaN, vary=true, min=-Inf, max=Inf, expr="", user_data=nothing)

    if isnan(value) && expr == ""
        error("One one of value or expr can be specified")
    end

    if !isnan(value) && expr != ""
        error("Either value or expr must be specified")
    end

    if vary && expr != ""
        error("Cannot vary and specify an expression")
    end

    return Parameter(name, value, vary, min, max, expr, user_data)
end
function Base.show(io::IO, p::Parameter) 
    println(io, String(p))
end
Base.String(p::Parameter) = "name=\"$(p.name)\",\tvalue=$(p.value),\tvary=$(p.vary),\tmin=$(p.min),\tmax=$(p.max),\texpr=$(p.expr),\tuser_data=$(p.user_data)"

struct Parameters
    parameters::Dict{String, Parameter}
end
Parameters() = Parameters(Dict{String, Parameter}())
Parameters(args...; kwargs...) = add!(Parameters(), args...; kwargs...)

function add!(ps::Parameters, p::Parameter)
    ps.parameters[p.name] = p
    ps
end
function add!(ps::Parameters, name::String; kwargs...)
    ps.parameters[name] = Parameter(name; kwargs...)
    ps
end
function add!(ps::Parameters, pvec::Vector{Parameter})
    for p in pvec
        add!(ps, p)
    end
    ps
end
function Base.getindex(ps::Parameters, indices...)
    ps.parameters[indices...]
end
Base.iterate(ps::Parameters) = iterate(ps.parameters)
Base.iterate(ps::Parameters, state) = iterate(ps.parameters, state)
Base.keys(ps::Parameters) = keys(ps.parameters)


function Base.show(io::IO, ps::Parameters)
    println(io, "Parameters:")
    for p in values(ps.parameters)
        println(io, "\t$(String(p))")
    end
end

"""
Model

Wraps a function of some paramters into something of the form that a fitter will accept

Owing to the multiple dispatch nature of Julia, argument names cannot be identified without a function call.

var_names are the names of the independent variables
"""
mutable struct Model
    name::String
    func::Function
    arg_names::Vector{String} # args in order
    kwarg_names::Vector{String} # kwargs in order
    var_names::OrderedDict{String, Int} # Independent variables (in order), pointing to their location in arg_names
    param_names::OrderedDict{String, Int} # Parameters (in order), pointing to their location in arg_names
end
function Model(name::String, func::Function, arg_names::Vector{String}, kwarg_names::Vector{String})

    if !allunique(kwarg_names)
        error("kwarg_names must be unique")
    end

    if !allunique(arg_names)
        error("arg_names must be unique")
    end

    param_names = OrderedDict(k => v for (v, k) in enumerate(arg_names))

    m = Model(name, func, arg_names, kwarg_names, OrderedDict{String, Int}(), param_names)

    # to reuse logic we now update the var_names
    update_vars!(m, arg_names[1:1])
end
"""
Allow the model to be evaluated given a set of parameters

x is a vector/tuple/whatever of independent variables
"""
function (m::Model)(xs, ps::Parameters)

    if length(xs) != length(m.var_names)
        error("xs must be same length as Paramerers.var_names")
    end

    args = Vector{Any}(undef, length(m.arg_names))
    for (name, p) in ps
        index = m.param_names[name]
        args[index] = p.value
    end

    for ((_, index), x) in zip(m.var_names, xs)
        args[index] = x
    end
    m.func(args...)
end


"""
update_vars!(m::Model, var_names)

Update the independent variables of the model
"""
function update_vars!(m::Model, var_names)

    if !allunique(var_names)
        error("var_names must be unique")
    end

    if !(var_names ⊆ m.arg_names)
        error("var_names must be a subset of arg_names")
    end

    var_indices = indexin(m.arg_names, var_names)
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
    arg_names::Vector{String} = [isa(arg, Expr) ? string(arg.args[1]) : string(arg) for arg in args]

    # because kwargs looks like Any[:($(Expr(:kw, :offset, 0.0))), :($(Expr(:kw, :cat, 2.0)))])
    kwarg_names::Vector{String} = [isa(arg, Expr) ? string(arg.args[1]) : string(arg) for arg in kwargs]

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
        param_symbol = Symbol(param)
        value = haskey(kwargs, param_symbol) ? kwargs[param_symbol] : -Inf
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
end


#
# Main code logic
#



end
