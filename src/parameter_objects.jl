"""
    Parameter

All the logic for the individual parameter class
"""
module ParameterObjects

    export AbstractParameter, Constant, Parameter, Expression, IndependentVariable
    export validate, depends_on
    export PARAMETERS

    """
    Parameters are the elemental components of the `LMfit` package.

    In order to improve my Julia programming, I am going to define the different kinds of possible parameters 
    in terms of a base class
    """
    abstract type AbstractParameter end
    function AbstractParameter(name; kwargs...) # dispatch to the desired type of parameter

        if get(kwargs, :independent, false) == true
            return IndependentVariable(name;)
        else
            return Parameter(name; kwargs...)
        end
    end
    validate(p::AbstractParameter) = nothing
    depends_on(p::AbstractParameter) = Set{Symbol}()
    

    function Base.show(io::IO, p::AbstractParameter) 
        println(io, String(p))
    end

    """
    Constant
    """
    mutable struct Constant{T} <: AbstractParameter
        name::Symbol
        value::T
    end
    Constant(name::Symbol; value=NaN) = Constant(name, value)

    Base.String(p::Constant) = "Constant: name=$(p.name),\tvalue=$(p.value)"


    """
    Parameter

    This is the elemental component of the `LMfit` package.

    The type T has to support Nan, Inf, and -Inf
    """
    mutable struct Parameter{T} <: AbstractParameter
        name::Symbol
        value::T
        min::T
        max::T
    end
    Parameter(name::Symbol; value=NaN, min=-Inf, max=Inf) = Parameter(name, value, min, max)

    Base.String(p::Parameter) = "Parameter: name=$(p.name),\tvalue=$(p.value),\tmin=$(p.min),\tmax=$(p.max)"

    function validate(p::Parameter)
        if p.min >= p.max
            error("item $(name): p.min must be less than p.max")
        end

        if p.value >= p.max || p.value <= p.min
            error("item $(name): p.value must be between p.min and p.max")
        end
    end


    """
    Expression

    This is the elemental component of the `LMfit` package.

    The type T has to support Nan, Inf, and -Inf
    """
    mutable struct Expression{T} <: AbstractParameter
        name::Symbol
        value::T
        min::T
        max::T
        expr::Expr
    end
    Expression(name::Symbol; expr=:(), value=NaN, min=-Inf, max=Inf) = return Expression(name, value, min, max, expr)

    Base.String(p::Expression) = "Expression: name=$(p.name),\tvalue=$(p.value),\tmin=$(p.min),\tmax=$(p.max),\texpr=$(p.expr)"

    function validate(p::Expression)
        if p.min >= p.max
            error("item $(name): p.min must be less than p.max")
        end
    end

    depends_on(p::Expression) = _get_symbols(p.expr)
  
    """
    IndependentVariable

    Used to identify independent variables
    """
    mutable struct IndependentVariable <: AbstractParameter
        name::Symbol
    end
    function Base.show(io::IO, p::IndependentVariable) 
        println(io, String(p))
    end
    Base.String(p::IndependentVariable) = "IndependentVariable: name=$(p.name)"

    function _get_symbols(ex)
        list = []
        walk!(list) = ex -> begin
           ex isa Symbol && push!(list, ex)
           ex isa Expr && ex.head == :call && map(walk!(list), ex.args[2:end])
           list
        end
        Set{Symbol}(walk!([])(ex))
    end

    # Annoying that I have to define this at the end of the module
    PARAMETERS = Dict(
        :constant=>Constant, 
        :expression=>Expression, 
        :parameter=>Parameter
        )

    # Conversion tools
    Constant(p::Parameter) = Constant(p.name, p.value)
    Constant(p::Expression) = Constant(p.name, p.value)

    Parameter(p::Constant; kwargs...) = Parameter(p.name; value=p.value, kwargs...)
    Parameter(p::Expression; kwargs...) = Parameter(p.name; value=p.value, min=p.min, max=p.max, kwargs...)

    Expression(p::Constant; kwargs...) = Expression(p.name; value=p.value, min=p.min, max=p.max, kwargs...)
    Expression(p::Parameter; kwargs...) = Expression(p.name; value=p.value, kwargs...)
end