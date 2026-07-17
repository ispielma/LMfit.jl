@testset "DSF-style fit patterns" begin
    function contextual_curve(times, c0, scale; a)
        return @. a.gain * (c0 * times + scale)
    end

    times = collect(range(0.0, 2.0, length=9))
    ctx = (gain=1.5,)
    scale = @. 0.7 * exp(-times * 0.4) + 0.2
    y = contextual_curve(times, 1.1, scale; a=ctx)
    inv_var = collect(range(0.5, 1.5, length=length(y)))

    model = Model(
        contextual_curve,
        [:times, :c0, :scale];
        var_names=[:times],
        kwarg_names=[:a],
    )
    params = Parameters()
    add!(
        params,
        :scale,
        :expression;
        expr=:(height .* exp.(-tvals .* gamma) .+ offset),
        value=ones(length(times)),
    )
    add!(params, :c0; value=0.8, min=0.0, max=2.0)
    add!(params, :tvals, :constant; value=times)
    add!(params, :height; value=0.5, min=0.0, max=2.0)
    add!(params, :offset, :constant; value=0.2)
    add!(params, :gamma; value=0.2, min=0.0, max=2.0)
    initial = Parameters(params)

    result = fit(
        model,
        y,
        inv_var,
        params;
        times,
        kwargs=Dict(:a => ctx),
    )

    @test model(result.ps_best; times, a=ctx) ≈ y atol=1.0e-8
    @test weights(result) == inv_var
    @test params[:c0].value == initial[:c0].value
    @test params[:scale].value == initial[:scale].value
    @test length(result.ps_best[:scale].value) == length(times)
    @test length(eval_uncertainty(result; times)) == length(times)
    @test all(isfinite, stderror(result))
end

@testset "Paper-style derived and constant parameters" begin
    model = @LMfit.generate_model function derived_curve(
        x,
        amp1,
        amp2,
        amp3,
        bias,
    )
        return @. amp1 * x + amp2 * x^2 + amp3 * x^3 + bias
    end
    LMfit.update_vars!(model, :x)

    x = range(-1.0, 1.0, length=17)
    y = @. 1.25 * x + 0.625 * x^2 + 0.3125 * x^3 + 0.1

    params = LMfit.Parameters()
    LMfit.add!(params, :amp1; value=1.0, min=0.0, max=2.0)
    LMfit.add!(params, :scale2; value=1.0)
    LMfit.add!(params, :scale2, :constant; value=0.5)
    LMfit.add!(params, :amp2, :expression; expr=:(scale2 .* amp1))
    LMfit.add!(params, :scale3, :constant; value=0.25)
    LMfit.add!(params, :amp3, :expression; expr=:(scale3 .* amp1 .^ power))
    LMfit.add!(params, :power, :constant; value=1.0)
    LMfit.add!(params, :bias, :constant; value=0.1)

    result = LMfit.fit(model, y, params; x)

    @test model(result.ps_best; x) ≈ y atol=1.0e-10
    @test result(; x) ≈ y atol=1.0e-10
    @test result.ps_best.parameters[:amp1].value ≈ 1.25 atol=1.0e-10
    @test result.ps_best[:amp2].value ≈ 0.625 atol=1.0e-10
    @test result.ps_best[:amp3].value ≈ 0.3125 atol=1.0e-10
    @test result.ps_best[:scale2].value == 0.5
    @test length(coef(result)) == 1
end

@testset "vector parameters with multidimensional output" begin
    model = @LMfit.generate_model function grid_curve(x, groups, amp, offsets)
        z = zeros(length(x), length(groups))
        for j in eachindex(groups)
            for i in eachindex(x)
                z[i, j] = amp * x[i] + offsets[j]
            end
        end
        return z
    end
    LMfit.update_vars!(model, :x, :groups)

    x = range(-1.0, 1.0, length=7)
    groups = 1:3
    y = grid_curve(x, groups, 1.75, [0.5, -0.25, 1.0])
    wt = ones(size(y))
    wt[1, 1] = 0.25

    params = Parameters()
    add!(params, :amp; value=1.0, min=0.0, max=3.0)
    add!(
        params,
        :offsets;
        value=[0.0, 0.0, 0.0],
        min=[-2.0, -2.0, -2.0],
        max=[2.0, 2.0, 2.0],
    )

    fit_result = fit(model, y, wt, params; x, groups)

    @test model(fit_result.ps_best; x, groups) ≈ y atol=1.0e-10
    @test size(residuals(fit_result)) == size(y)
    @test weights(fit_result) == wt
    @test fit_result.ps_best[:amp].value ≈ 1.75 atol=1.0e-10
    @test fit_result.ps_best[:offsets].value ≈ [0.5, -0.25, 1.0] atol=1.0e-10
    @test length(fit_result.ps_best[:offsets].σ) == 3
end

@testset "fit without independent variables" begin
    function context_curve(amp; a)
        return @. a.bias + amp * a.x
    end

    ctx = (x=range(0.0, 2.0, length=8), bias=0.3)
    y = context_curve(2.25; a=ctx)
    wt = collect(range(1.0, 2.0, length=length(y)))
    model = Model(context_curve, [:amp]; var_names=[], kwarg_names=[:a])
    params = Parameters(:amp; value=1.0, min=0.0, max=4.0)

    result = fit(model, y, wt, params; kwargs=Dict(:a => ctx))

    @test result.ps_best[:amp].value ≈ 2.25 atol=1.0e-10
    @test result.fm(result |> coef) ≈ y atol=1.0e-10
    @test weights(result) == wt
end

@testset "declared model keyword routing" begin
    model = @generate_model direct_context(x, slope; gain=1.0) =
        @. gain * slope * x
    update_vars!(model, :x)

    x = collect(range(0.0, 2.0, length=8))
    y = direct_context(x, 1.75; gain=2.0)
    params = make_params(model; slope=1.0)

    result = fit(model, y, params; x, gain=2.0)

    @test result.ps_best[:slope].value ≈ 1.75 atol=1.0e-10
    @test model(result.ps_best; x, gain=2.0) ≈ y atol=1.0e-10
end
