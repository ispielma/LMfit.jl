"""
    ParameterObjects

All the logic for the individual parameter class
"""
module ParameterObjects

    using OrderedCollections
    using SpecialFunctions # make special functions available to the fit
    # TODO this suggests it might be smart to move the evaluator function
    # Into its own module to keep the name space as clean as possible

    export AbstractParameter, Constant, Parameter, Expression, IndependentVariable
    export validate, depends_on, _update_params_from_vect!
    export PARAMETERS

    """
    Parameters are the elemental components of the `LMfit` package.

    In order to improve my Julia programming, I am going to define the different kinds of possible parameters 
    in terms of an abstract type
    """
    abstract type AbstractParameter end
    AbstractParameter(p::AbstractParameter) = deepcopy(p)

    function AbstractParameter(name; kwargs...) # dispatch to the desired type of parameter

        if get(kwargs, :independent, false) == true
            return IndependentVariable(name;)
        else
            return Parameter(name; kwargs...)
        end
    end
    validate(p::AbstractParameter) = nothing
    depends_on(p::AbstractParameter) = Set{Symbol}()
    Base.length(p::AbstractParameter) = length(p.value)

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

    function Base.String(p::Constant{T}) where T
        s = _sep(T)
        "Constant: $(p.name)$(s)value=$(p.value)"
    end
    """
    AbstractIndependentParameter

    This is a second abstract type for quantities that are directly varied

    This exists so I can create a subtype that also includes incertanties
    """
    abstract type  AbstractIndependentParameter <: AbstractParameter end

    function validate(p::AbstractIndependentParameter)
        if length(p) != length(p.min) || length(p) != length(p.max)
            error("item $(p.name): length of all values and limit variables must be the same")
        end

        if any(min > max for (min, max) in zip(p.min, p.max))
            error("item $(p.name): p.min = $(p.min) must be less than p.max = $(p.max)")
        end

        if any((value > max || value < min) for (min, max, value) in zip(p.min, p.max, p.value))
            error("item $(p.name): p.value = $(p.value) must be between p.min=$(p.min) and p.max=$(p.max)")
        end
        true
    end

    """
    Parameter

    This is the elemental component of the `LMfit` package.

    The type T has to support Nan, Inf, and -Inf
    """
    mutable struct Parameter{T} <: AbstractIndependentParameter
        name::Symbol
        value::T
        min::T
        max::T
    end
    function Parameter(name::Symbol; value=NaN, min=nothing, max=nothing)
        if max === nothing
            max = typemax.(value)
        end

        if min === nothing
            min = typemin.(value)
        end

        Parameter(name, value, min, max)
    end
    function Parameter(name::Symbol, arg)
        println("Parameter: $(name),\targ=$(arg)")
    end

    function Base.String(p::Parameter{T}) where T
        s = _sep(T)
        "Parameter: $(p.name)$(s)value=$(p.value)$(s)min=$(p.min)$(s)max=$(p.max)"
    end
    """
    ParameterWithUncertanty

    After performing a fit model parameters are replaced with this to show the fit untertanties
    """
    mutable struct ParameterWithUncertanty{T} <: AbstractIndependentParameter
        name::Symbol
        value::T
        min::T
        max::T
        σ::T
    end
    function ParameterWithUncertanty(name::Symbol; value=NaN, min=nothing, max=nothing, σ=nothing)
        if max === nothing
            max = typemax.(value)
        end

        if min === nothing
            min = typemin.(value)
        end

        if σ === nothing
            σ = zero(value)
        end

        ParameterWithUncertanty(name, value, min, max, σ)
    end
    function Base.String(p::ParameterWithUncertanty{T}) where T
        s = _sep(T)
        "Parameter with uncertainty: $(p.name)$(s)value=$(p.value)$(s)σ=$(p.σ)$(s)min=$(p.min)$(s)max=$(p.max)"
    end

    """
    Expression

    This is the elemental component of the `LMfit` package.

    The type T has to support Nan, Inf, and -Inf
    """
    mutable struct Expression{T} <: AbstractParameter
        name::Symbol
        value::T
        expr::Expr
    end
    Expression(name::Symbol; expr=:(), value=NaN) = return Expression(name, value, expr)

    function Base.String(p::Expression{T}) where T
        s = _sep(T)
        "Expression: $(p.name)\texpr=$(p.expr)$(s)value=$(p.value)"
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
    Base.String(p::IndependentVariable) = "IndependentVariable: $(p.name)"

    function _get_symbols(ex)
        symbols = Symbol[]
        _collect_symbols!(symbols, ex)
        return Set(symbols)
    end

    function _collect_symbols!(symbols, ex)
        if ex isa Symbol
            push!(symbols, ex)
        elseif ex isa Expr
            if ex.head == :. && !(ex.args[2] isa Expr && ex.args[2].head == :tuple)
                _collect_symbols!(symbols, ex.args[1])
                return symbols
            end

            first_arg = ex.head in (:call, :.) ? 2 : 1
            if first_arg <= length(ex.args)
                for i in first_arg:length(ex.args)
                    _collect_symbols!(symbols, ex.args[i])
                end
            end
        end
        return symbols
    end

    # Annoying that I have to define this at the end of the module
    PARAMETERS = Dict(
        :constant=>Constant, 
        :expression=>Expression, 
        :parameter=>Parameter
        )

    # Conversion tools
    Constant(p::AbstractIndependentParameter) = Constant(p.name, p.value)
    Constant(p::Expression) = Constant(p.name, p.value)

    Parameter(p::Constant; kwargs...) = Parameter(p.name; value=p.value, kwargs...)
    Parameter(p::AbstractIndependentParameter; kwargs...) = Parameter(p.name; value=p.value, kwargs...)
    Parameter(p::Expression; kwargs...) = Parameter(p.name; value=p.value, min=p.min, max=p.max, kwargs...)

    ParameterWithUncertanty(p::Constant; kwargs...) = ParameterWithUncertanty(p.name; value=p.value, kwargs...)
    ParameterWithUncertanty(p::AbstractIndependentParameter; kwargs...) = ParameterWithUncertanty(p.name; value=p.value, kwargs...)
    ParameterWithUncertanty(p::Expression; kwargs...) = ParameterWithUncertanty(p.name; value=p.value, min=p.min, max=p.max, kwargs...)

    Expression(p::Constant; kwargs...) = Expression(p.name; value=p.value, min=p.min, max=p.max, kwargs...)
    Expression(p::AbstractIndependentParameter; kwargs...) = Expression(p.name; value=p.value, kwargs...)

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
    Parameters(ps::Parameters) = deepcopy(ps)
    Parameters() = Parameters(OrderedDict{Symbol, Parameter}())
    Parameters(args...; kwargs...) = add!(Parameters(), args...; kwargs...)

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
    Base.setindex!(ps::Parameters, value, key) = setindex!(ps.parameters, value, key)

    function Base.show(io::IO, ps::Parameters)
        println(io, "Parameters:")
        for p in values(ps.parameters)
            println(io, "\t$(String(p))")
        end
    end

    # New methods

    function add!(ps::Parameters, p::AbstractParameter)
        ps.parameters[p.name] = p
        ps
    end
    function add!(ps::Parameters, pvec::Vector{AbstractParameter})
        for p in pvec
            add!(ps, p)
        end
        ps
    end
    add!(ps::Parameters, name::Symbol, parameter_type::Symbol, args...; kwargs...) = add!(ps, PARAMETERS[parameter_type](name, args...; kwargs...))
    function add!(ps::Parameters, name::Symbol, args...; kwargs...)

        kwargs = Dict(kwargs)
        param_type = pop!(kwargs, :param_type, :parameter) # default to a parameter, otherwise the keyword argument :param_type will be used
        add!(ps::Parameters, name::Symbol, param_type, args...; kwargs...) 
    end
    """
        depends_on

    find all dependencies between parameters
    """
    depends_on(ps::Parameters) = Dict(k=>depends_on(p) for (k, p) in ps)

    """
        validate

    check for error conditions
    """
    function validate(ps::Parameters)
        for (name, p) in ps
            if name != p.name
                error("item $(name): Parameters name-key must match name-field the associated record")
            end

            validate(p)
        end
        return true
    end

    """
        find_dependencies!(ps::Parameters)

    This is a key part of the logic of this package.  It will iterate over the parameters to see if they are
    fully determined, and to check for error conditions such as circular dependencies.

    It will then resort the parameters in the order that they need to be resolved if there are no errors
    """
    function find_dependencies!(ps::Parameters)

        # Find dependencies
        dependencies = depends_on(ps)

        sorted_parameters = empty(ps.parameters) # create an empty version of ps.parameters

        resolved_one = true
        while resolved_one 
            resolved_one = false

            # find resolved dependencies
            for (name, dep) in dependencies
                if isempty(dep)
                    sorted_parameters[name] = ps[name]
                    delete!(dependencies, name)
                    resolved_one = true
                end
            end

            # remove resolved variables from depends_on sets
            for dep in values(dependencies)
                for sorted in values(sorted_parameters)
                    delete!(dep, sorted.name)
                end
            end
        end

        if !isempty(dependencies)
            error("Circular dependencies detected")
        end

        empty!(ps.parameters)
        for (name, p) in sorted_parameters
            ps.parameters[name] = p
        end

        return ps
    end

    """
        resolve_parameters(ps::Parameters)

    Build the in-process parameter resolver used by the fitting adapter. The
    returned function accepts the flat vector varied by LsqFit and returns all
    parameter values in dependency order as a flat `Vector{Float64}`.
    """
    function resolve_parameters(ps::Parameters)
        ordered = collect(values(ps))
        return params -> _resolve(ordered, params)
    end

    function _resolve(ordered, params)
        env = Dict{Symbol, Any}()
        index = 1

        for p in ordered
            if p isa AbstractIndependentParameter
                last = index + length(p) - 1
                env[p.name] = params[index:last]
                index = last + 1
            elseif p isa Constant
                env[p.name] = p.value
            elseif p isa Expression
                env[p.name] = _eval_expr(p.expr, env)
            end
        end

        result = Float64[]
        for p in ordered
            value = env[p.name]
            if value isa Number
                push!(result, value)
            else
                append!(result, value)
            end
        end
        return result
    end

    _eval_expr(value, env) = value
    _eval_expr(value::QuoteNode, env) = value.value

    function _eval_expr(name::Symbol, env)
        if haskey(env, name)
            return env[name]
        elseif isdefined(@__MODULE__, name)
            return getfield(@__MODULE__, name)
        elseif isdefined(Base, name)
            return getfield(Base, name)
        end
        throw(UndefVarError(name))
    end

    function _eval_expr(ex::Expr, env)
        if ex.head == :call
            return _eval_call(ex.args, env)
        elseif ex.head == :.
            if ex.args[2] isa Expr && ex.args[2].head == :tuple
                fn = _eval_expr(ex.args[1], env)
                args = _eval_expr(ex.args[2], env)
                return broadcast(fn, args...)
            end
            value = _eval_expr(ex.args[1], env)
            name = _eval_expr(ex.args[2], env)
            return getproperty(value, name)
        elseif ex.head == :tuple
            return Tuple(_eval_expr(arg, env) for arg in ex.args)
        elseif ex.head == :ref
            value = _eval_expr(ex.args[1], env)
            indices = Tuple(_eval_expr(arg, env) for arg in ex.args[2:end])
            return getindex(value, indices...)
        elseif ex.head == :vect
            return [_eval_expr(arg, env) for arg in ex.args]
        end
        throw(ArgumentError("unsupported parameter expression: $(ex)"))
    end

    function _eval_call(args, env)
        name = args[1]
        values = Any[]
        for arg in args[2:end]
            push!(values, _eval_expr(arg, env))
        end

        if name isa Symbol && startswith(String(name), ".")
            plain_name = Symbol(String(name)[2:end])
            fn = _eval_expr(plain_name, env)
            return broadcast(fn, values...)
        end

        fn = _eval_expr(name, env)
        return fn(values...)
    end

    """
        _update_params_from_vect!(ps::Parameters, params_vect)
    
    Update parameters from the entries provided in params_vect

    keys defines which parameters to update and aligns the order of the params_vect

    When passed only a Parameters object,  assumes that the parameters are ordered in the same way as the params_vect
    """
    function _update_params_from_vect!(ps::Parameters, field::Symbol, keys, params_vect)
        index = 1
        for k in keys
            p = ps[k]
            if typeof(getfield(p, field)) <: Number
                len = 1
                setfield!(p, field, params_vect[index])
            else
                len = length(p)
                setfield!(p, field, params_vect[index:index+len-1])
            end
            index += len
        end
        ps
    end
    _update_params_from_vect!(ps::Parameters, field::Symbol, vect) = _update_params_from_vect!(ps, field, keys(ps), vect)

    _sep(T::Type) = T <: Number ? "\t" : "\n\t\t"


end
