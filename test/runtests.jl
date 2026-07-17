using Test
using LMfit

include("demo_fits.jl")
include("consumer_patterns.jl")

@testset "confidence bands" begin
    line(x, a, b) = @. a * x + b
    model = Model(line, [:x, :a, :b]; var_names=[:x])
    params = make_params(model; a=0.0, b=0.0)
    x = collect(0.0:4.0)
    y = [1.0, 3.2, 4.8, 7.1, 8.9]
    result = fit(model, y, params; x)
    fitted = copy(coef(result))

    band = eval_uncertainty(result; x=range(0.0, 4.0, length=9))

    @test length(band) == 9
    @test all(isfinite, band)
    @test coef(result) == fitted
end

@testset "confidence-band failures preserve coefficients" begin
    fail = Ref(false)
    limit = Ref(Inf)
    function guarded_line(x, a, b)
        fail[] && a > limit[] && error("perturbed evaluation failed")
        @. a * x + b
    end

    model = Model(guarded_line, [:x, :a, :b]; var_names=[:x])
    params = make_params(model; a=0.0, b=0.0)
    x = collect(0.0:4.0)
    y = [1.0, 3.2, 4.8, 7.1, 8.9]
    result = fit(model, y, params; x)
    fitted = copy(coef(result))
    limit[] = fitted[1] + 0.005 * stderror(result)[1]
    fail[] = true

    @test_throws ErrorException eval_uncertainty(result; x)
    @test coef(result) == fitted
end

@testset "default unit inverse variance" begin
    line(x, a, b) = @. a * x + b
    model = Model(line, [:x, :a, :b]; var_names=[:x])
    params = make_params(model; a=0.0, b=0.0)
    x = collect(0.0:4.0)
    y = [1.0, 3.2, 4.8, 7.1, 8.9]

    default = fit(model, y, params; x)
    unit = fit(model, y, params; x, covar_inv=ones(length(y)))

    @test coef(default) ≈ coef(unit)
    @test stderror(default) ≈ stderror(unit)
end

@testset "covariance scaling" begin
    line(x, a, b) = @. a * x + b
    model = Model(line, [:x, :a, :b]; var_names=[:x])
    params = make_params(model; a=0.0, b=0.0)
    x = collect(0.0:4.0)
    y = [1.0, 3.2, 4.8, 7.1, 8.9]

    err = try
        fit(model, y, params; x, scale_covar=true)
        nothing
    catch err
        err
    end

    @test err isa ErrorException
    @test sprint(showerror, err) == "scale_covar not implemented yet"
end
