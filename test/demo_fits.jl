using LazyGrids
using Random

@testset "demo fit results" begin
    Random.seed!(20240717)

    gaussian(x, amp, cen, wid) = @. amp * exp(-(x - cen)^2 / (2 * wid^2))

    x = range(-10, 10, length=101)
    y = gaussian(x, 2.33, 0.21, 1.51)
    y_noise = y .+ 0.5 .* randn(size(x))

    model = Model(gaussian, [:x, :amp, :cen, :wid]; var_names=[:x])
    params = Parameters()
    add!(params, :amp; value=1.0, min=0.0, max=5.0)
    add!(params, :cen; value=0.0, min=-5.0, max=5.0)
    add!(params, :wid; value=1.0)
    result = fit(model, y_noise, params; x)

    @test coef(result) == [
        1.3889647300069357,
        2.382008745535082,
        0.3280366591457827,
    ]
    @test [
        result.ps_best[:amp].value,
        result.ps_best[:cen].value,
        result.ps_best[:wid].value,
    ] == [
        2.382008745535082,
        0.3280366591457827,
        1.3889647300069357,
    ]

    model = @generate_model gaussian_generated(x, amp, cen, wid) =
        @. amp * exp(-(x - cen)^2 / (2 * wid^2))
    update_vars!(model, :x)
    params = make_params(
        model;
        amp=(value=1.0, min=0.0, max=5.0),
        cen=(value=0.0, min=-5.0, max=5.0),
        wid=1.0,
    )
    result = fit(model, y_noise, params; x)

    @test coef(result) == [
        1.3889647300069357,
        2.382008745535082,
        0.3280366591457827,
    ]
    @test [
        result.ps_best[:amp].value,
        result.ps_best[:cen].value,
        result.ps_best[:wid].value,
    ] == [
        2.382008745535082,
        0.3280366591457827,
        1.3889647300069357,
    ]

    model = @generate_model gaussian_constrained(x, amp, cen, wid) =
        @. amp * exp(-(x - cen)^2 / (2 * wid^2))
    update_vars!(model, :x)
    params = make_params(model; amp=2.33, cen=0.21, wid=1.51)
    params[:wid] = Expression(params[:wid]; expr=:(cen .* amp))
    result = fit(model, y_noise, params; x)

    @test coef(result) == [2.528276027214923, 0.5065977110241855]
    @test [
        result.ps_best[:amp].value,
        result.ps_best[:cen].value,
        result.ps_best[:wid].value,
    ] == [
        2.528276027214923,
        0.5065977110241855,
        1.2808188482244014,
    ]

    model = @generate_model gaussian2D_scalar(x, y, amp, cen_x, cen_y, wid_x, wid_y) =
        @. amp * exp(
            -(x - cen_x)^2 / (2 * wid_x^2) -
            (y - cen_y)^2 / (2 * wid_y^2)
        )
    x = range(-10, 10, length=101)
    y = range(-10, 10, length=101)
    xgrid, ygrid = ndgrid(x, y)
    z = gaussian2D_scalar(xgrid, ygrid, 2.33, 0.21, -0.5, 1.51, 2.5)
    z_noise = z .+ 0.5 .* randn(size(xgrid))
    update_vars!(model, :x, :y)
    params = Parameters()
    add!(params, :amp; value=1.0, min=0.0, max=5.0)
    add!(params, :cen_x; value=0.0, min=-5.0, max=5.0)
    add!(params, :cen_y; value=0.0, min=-5.0, max=5.0)
    add!(params, :wid_x; value=1.0)
    add!(params, :wid_y; value=1.0)
    result = fit(model, z_noise, params; x=xgrid, y=ygrid)

    @test coef(result) == [
        -0.41925068139840954,
        2.4849162161371017,
        1.5382927653197986,
        0.23134686330092782,
        2.3002772829711433,
    ]
    @test [
        result.ps_best[:amp].value,
        result.ps_best[:cen_x].value,
        result.ps_best[:cen_y].value,
        result.ps_best[:wid_x].value,
        result.ps_best[:wid_y].value,
    ] == [
        2.3002772829711433,
        0.23134686330092782,
        -0.41925068139840954,
        1.5382927653197986,
        2.4849162161371017,
    ]
    @test size(residuals(result)) == (101, 101)

    model = @generate_model gaussian2D_vector(x, y, amp, cen, wid) =
        @. amp * exp(
            -(x - cen[1])^2 / (2 * wid[1]^2) -
            (y - cen[2])^2 / (2 * wid[2]^2)
        )
    x = range(-10, 10, length=101)
    y = range(-10, 10, length=101)
    xgrid, ygrid = ndgrid(x, y)
    z = gaussian2D_vector(xgrid, ygrid, 2.33, [0.21, -0.5], [1.51, 2.5])
    z_noise = z .+ 0.5 .* randn(size(xgrid))
    update_vars!(model, :x, :y)
    params = Parameters()
    add!(params, :amp; value=1.0, min=0.0, max=5.0)
    add!(
        params,
        :cen;
        value=[0.0, 0.0],
        min=[-5.0, -5.0],
        max=[5.0, 5.0],
    )
    add!(params, :wid; value=[1.0, 1.0])
    result = fit(model, z_noise, params; x=xgrid, y=ygrid)

    @test coef(result) == [
        1.4839026261160218,
        2.4668175655904867,
        2.3195388315108807,
        0.2127827589650404,
        -0.49246055747891404,
    ]
    @test vcat(
        result.ps_best[:amp].value,
        result.ps_best[:cen].value,
        result.ps_best[:wid].value,
    ) == [
        2.3195388315108807,
        0.2127827589650404,
        -0.49246055747891404,
        1.4839026261160218,
        2.4668175655904867,
    ]
    @test size(residuals(result)) == (101, 101)
end
