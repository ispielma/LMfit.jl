"""
    ModelObjects

All the logic for the model class
"""
module ModelObjects
    using OrderedCollections
    using ExprTools

    import ..ParameterObjects: Parameter, Constant, Expression, IndependentVariable, Parameters
    import ..ParameterObjects: validate, add!, find_dependencies!, resolve_parameters

    """
    Model

    Wraps a function of some paramters into something of the form that a fitter will accept

    Owing to the multiple dispatch nature of Julia, argument names cannot be identified without a function call.

    var_names are the names of the independent variables
    """
    mutable struct Model
        name::Symbol
        func::Function
        arg_names::Vector{Symbol} # args in order
        kwarg_names::Vector{Symbol} # kwargs in order
        var_names::OrderedDict{Symbol, Int} # Independent variables (in order), pointing to their location in arg_names
        param_names::OrderedDict{Symbol, Int} # Parameters (in order), pointing to their location in arg_names
    end
    function Model(func::Function, arg_names; kwarg_names=[], var_names=[])

        if !allunique(kwarg_names)
            error("kwarg_names must be unique")
        end

        if !allunique(arg_names)
            error("arg_names must be unique")
        end

        arg_names = [Symbol(name) for name in arg_names]
        kwarg_names = [Symbol(name) for name in kwarg_names]
        var_names = [Symbol(name) for name in var_names]

        param_names = OrderedDict(k => v for (v, k) in enumerate(arg_names))

        m = Model(Symbol(func), func, arg_names, kwarg_names, OrderedDict{Symbol, Int}(), param_names)

        # to reuse logic we now update the var_names
        update_vars!(m, var_names...)
    end

    function Base.show(io::IO, m::Model)
        println(io, "Model: $(m.name)")
        print(io, "\tFunction arguments: ")

        for name in m.arg_names
            print(io, "$(name), ")
        end

        if length(m.kwarg_names) > 0
            print(io, "\n\tFunction keyword arguments: ")
            for name in m.kwarg_names
                print(io, "$(name), ")
            end
        end

        if length(m.var_names) > 0
            print(io, "\n\tIndependent variables: ")
            for name in keys(m.var_names)
                print(io, "$(name), ")
            end
        end

        if length(m.param_names) > 0
            print(io, "\n\tParameters: ")
            for name in keys(m.param_names)
                print(io, "$(name), ")
            end
        end
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

        (vars, not_vars) = _strip_vars_kwargs(m, kwargs)

        if length(p) != length(m.param_names)
            error("p must be same length as Model.param_names")
        end

        args = Vector{Any}(undef, length(m.arg_names))
        for ( (_, index), param) in zip(m.param_names, p)
            args[index] = param
        end

        # Get independent variables
        for (name, index) in m.var_names
            args[index] = vars[name]
        end

        m.func(args...; not_vars...)
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
            Model($name, $arg_names; kwarg_names=$kwarg_names)
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

        if !(keys(kwargs) ⊆ keys(m.param_names))
            error(" kwargs = $(keys(kwargs)) must be a subset of keys(m.param_names) = $(keys(m.param_names)) ")
        end


        for param in keys(m.param_names)
            kwarg = haskey(kwargs, param) ? kwargs[param] : -Inf

            if typeof(kwarg) <: Union{AbstractDict, NamedTuple}
                add!(ps, param; kwarg...)
            else
                add!(ps, param; value=kwarg)
            end
        end

        return ps
    end

    """
    _strip_vars_kwargs(m::Model, kwargs)

    kwargs can contain a mixture of model variables and other flags, so we will strip them out

    returns split out named tuples ()
    """
    function _strip_vars_kwargs(m::Model, kwargs)

        if !(keys(m.var_names) ⊆ keys(kwargs))
            error("keys(m.var_names) = $(keys(m.var_names)) must be a subset of kwargs = $(keys(kwargs))")
        end

        vars = pairs(NamedTuple( k=>kwargs[k] for k in keys(m.var_names) ))
        not_vars = pairs(NamedTuple( k=>v for (k, v) in kwargs if !haskey(m.var_names, k) ))

        return (vars, not_vars)

    end
end