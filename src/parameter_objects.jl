"""
    Parameter

All the logic for the individual parameter class
"""
module ParameterObjects

export Parameter


    abstract type AbstractParameter end

    """
    Parameter

    This is the elemental component of the `LMfit` package.

    The type T has to support Nan, Inf, and -Inf
    """
    mutable struct Parameter{T} <: AbstractParameter
        name::Symbol
        value::T
        vary::Bool
        min::T
        max::T
        expr::Expr
        user_data::Any
        _depends_on::Set{Symbol} # Used internally mark what other paramteres the current parameter depends on
    end
    function Parameter(name::Symbol; value=NaN, vary=nothing, min=-Inf, max=Inf, expr=:(), user_data=nothing)

        if isnan(value) && expr == :()
            error("One one of value or expr can be specified")
        end

        if !isnan(value) && expr != :()
            error("Either value or expr must be specified")
        end

        if vary==true && expr != :()
            error("Cannot vary and specify an expression")
        end

        if isnothing(vary)
            vary = expr != :() ? false : true
        end

        return Parameter(name, value, vary, min, max, expr, user_data, Set{Symbol}())
    end
    function Base.show(io::IO, p::Parameter) 
        println(io, String(p))
    end
    Base.String(p::Parameter) = "name=$(p.name),\tvalue=$(p.value),\tvary=$(p.vary),\tmin=$(p.min),\tmax=$(p.max),\texpr=$(p.expr),\tuser_data=$(p.user_data)"

end