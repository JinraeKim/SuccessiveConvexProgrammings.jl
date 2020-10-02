using SuccessiveConvexProgrammings
const SCvxs = SuccessiveConvexifications
using Test

using LinearAlgebra
using Convex
const Cvx = Convex
using Debugger
using Plots
ENV["GKSwstype"]="nul"  # do not show plot


function print_message(name::String)
    println(">"^6 * name * "<"^6)
end

function test_initial_guess()
    print_message("initial_guess")
    n_x, n_u, N = 4, 2, 11
    scvx = SCvx(N=N, n_x=n_x, n_u=n_u)
    X_0 = scvx.X_k
    U_0 = scvx.U_k
    X_k = zeros(N, n_x)
    U_k = ones(N-1, n_u)
    initial_guess!(scvx, X_k, U_k)
    @test norm(scvx.X_k - X_0) == 0.0
    @test norm(scvx.U_k - U_0) != 0.0
end

################## get_J ##################
function test_get_J()
    print_message("get_J")
    n_x, n_u, N = 4, 2, 11
    scvx = SCvx(N=N, n_x=n_x, n_u=n_u,
                objs_path=[my_path_obj],
                objs_terminal=[my_terminal_obj],
                consts_path_ineq=[not_too_large_input],
                consts_initial_eq=[my_const_initial_eq],
                consts_terminal_eq=[my_const_terminal_eq],
               )
    X, U = scvx.X_k, scvx.U_k  # zeros
    obj = SCvxs.get_J(scvx, X, U)
    @test obj == (
                  16.0
                  + (N-1) * scvx.λ * 2.0 * n_u
                  + 0.0 + scvx.λ * norm(my_const_terminal_eq(X[end, :]), 1)
                 )
end

## get_obj
function test_get_obj()
    print_message("get_obj")
    n_x, n_u, N = 4, 2, 11
    scvx = SCvx(N=N, n_x=n_x, n_u=n_u,
                objs_path=[my_path_obj],
                objs_terminal=[my_terminal_obj],
               )
    X, U = scvx.X_k, scvx.U_k  # zeros
    obj = SCvxs.get_obj(scvx, X, U)
    @test obj == 16.0
end

function test_calculate_objs_path()
    print_message("calculate_objs_path")
    n_x, n_u, N = 4, 2, 11
    scvx = SCvx(N=N, n_x=n_x, n_u=n_u,
                objs_path=[my_path_obj],
               )
    X, U = scvx.X_k, scvx.U_k  # zeros
    path_obj = SCvxs.calculate_objs_path(scvx, X, U)
    @test path_obj == 0.0
end

function test_calculate_objs_terminal()
    print_message("calculate_objs_terminal")
    n_x, n_u, N = 4, 2, 11
    scvx = SCvx(N=N, n_x=n_x, n_u=n_u,
                objs_terminal=[my_terminal_obj],
               )
    X, U = scvx.X_k, scvx.U_k  # zeros
    terminal_obj = SCvxs.calculate_objs_terminal(scvx, X)
    @test terminal_obj == 16.0
end

## get_obj_const_penalty
function test_get_obj_const_penalty()
    print_message("get_obj_const_penalty")
    n_x, n_u, N = 4, 2, 11
    scvx = SCvx(N=N, n_x=n_x, n_u=n_u,
                consts_path_ineq=[not_too_large_input],
                consts_initial_eq=[my_const_initial_eq],
                consts_terminal_eq=[my_const_terminal_eq],
               )
    X, U = scvx.X_k, scvx.U_k  # zeros
    obj = SCvxs.get_obj_const_penalty(scvx, X, U)
    @test obj == (N-1) * scvx.λ * 2.0 * n_u + 0.0 + scvx.λ * norm(my_const_terminal_eq(X[end, :]), 1)
end

function test_calculate_const_path_penalty()
    print_message("calculate_const_path_penalty")
    n_x, n_u, N = 4, 2, 11
    scvx = SCvx(N=N, n_x=n_x, n_u=n_u,
                consts_path_ineq=[not_too_large_input],
               )
    X, U = scvx.X_k, scvx.U_k  # zeros
    obj = SCvxs.calculate_const_path_penalty(scvx, X, U)
    @test obj == (N-1) * scvx.λ * 2.0 * n_u
end

function test_calculate_const_initial_penalty()
    print_message("calculate_const_initial_penalty")
    n_x, n_u, N = 4, 2, 11
    scvx = SCvx(N=N, n_x=n_x, n_u=n_u,
                consts_initial_eq=[my_const_initial_eq],
               )
    X, U = scvx.X_k, scvx.U_k  # zeros
    obj = SCvxs.calculate_const_initial_penalty(scvx, X, U)
    @test obj == 0.0
end

function test_calculate_const_terminal_penalty()
    print_message("calculate_const_terminal_penalty")
    n_x, n_u, N = 4, 2, 11
    scvx = SCvx(N=N, n_x=n_x, n_u=n_u,
                consts_terminal_eq=[my_const_terminal_eq],
               )
    X, U = scvx.X_k, scvx.U_k  # zeros
    obj = SCvxs.calculate_const_terminal_penalty(scvx, X)
    @test obj == scvx.λ * norm(my_const_terminal_eq(X[end, :]), 1)
end

################## get_L ##################
function test_get_L()
    print_message("get_L")
    n_x, n_u, N = 4, 2, 11
    scvx = SCvx(N=N, n_x=n_x, n_u=n_u,
                objs_path=[my_path_obj],
                objs_terminal=[my_terminal_obj],
                consts_path_ineq=[not_too_large_input],
                consts_initial_eq=[my_const_initial_eq],
                consts_terminal_eq=[my_const_terminal_eq],
               )
    X, U = scvx.X_k, scvx.U_k  # zeros
    D, W = X, U
    jacob_dict = Dict()
    obj = SCvxs.get_L(scvx, D, W, jacob_dict)
    @test obj == (
                  16.0
                  + (N-1) * scvx.λ * 2.0 * n_u
                  + 0.0 + scvx.λ * norm(my_const_terminal_eq(X[end, :]), 1)
                 )
end

## get_obj_linearised
function test_get_obj_linearised()
    print_message("get_obj_linearised")
    n_x, n_u, N = 4, 2, 11
    scvx = SCvx(N=N, n_x=n_x, n_u=n_u,
                objs_path=[my_path_obj],
                objs_terminal=[my_terminal_obj],
               )
    X, U = scvx.X_k, scvx.U_k  # zeros
    obj, jacob_dict = SCvxs.get_obj_linearised(scvx, X, U)
    @test obj == 0.0 + 16.0
end

function test_calculate_objs_path_linearised()
    print_message("calculate_objs_path_linearised")
    n_x, n_u, N = 4, 2, 11
    scvx = SCvx(N=N, n_x=n_x, n_u=n_u,
                objs_path=[my_path_obj],
               )
    X, U = scvx.X_k, scvx.U_k  # zeros
    D, W = X, U  # originally, it should be values from Convex.Variable
    jacob_dict = Dict()
    obj, _ = SCvxs.calculate_objs_path_linearised(scvx, D, W, jacob_dict)
    @test obj == 0.0
end
function test_calculate_objs_terminal_linearised()
    print_message("calculate_objs_terminal_linearised")
    n_x, n_u, N = 4, 2, 11
    scvx = SCvx(N=N, n_x=n_x, n_u=n_u,
                objs_terminal=[my_terminal_obj],
               )
    X, U = scvx.X_k, scvx.U_k  # zeros
    D, W = X, U  # originally, it should be values from Convex.Variable
    jacob_dict = Dict()
    obj, _ = SCvxs.calculate_objs_terminal_linearised(scvx, D, W, jacob_dict)
    @test obj == 16.0
end

## get_obj_const_penalty_linearised
function test_get_obj_const_penalty_linearised()
    print_message("get_obj_const_penalty_linearised")
    n_x, n_u, N = 4, 2, 11
    scvx = SCvx(N=N, n_x=n_x, n_u=n_u,
                consts_path_ineq=[not_too_large_input],
                consts_path_eq=[my_const_path_eq],
                consts_initial_eq=[my_const_initial_eq],
                consts_terminal_eq=[my_const_terminal_eq],
               )
    X, U = scvx.X_k, scvx.U_k  # zeros
    obj, jacob_dict = SCvxs.get_obj_const_penalty_linearised(scvx, X, U)
    @test obj == (N-1) * scvx.λ * 2.0 * n_u + 0.0 + scvx.λ * 2.0 * n_x
end

function test_calculate_const_path_penalty_linearised()
    print_message("calculate_const_path_penalty_linearised")
    n_x, n_u, N = 4, 2, 11
    scvx = SCvx(N=N, n_x=n_x, n_u=n_u,
                consts_path_ineq=[not_too_large_input],
               )
    X, U = scvx.X_k, scvx.U_k  # zeros
    D, W = X, U  # originally, it should be values from Convex.Variable
    jacob_dict = Dict()
    obj, _ = SCvxs.calculate_const_path_penalty_linearised(scvx, D, W, jacob_dict)
    @test obj == (N-1) * scvx.λ * 2.0 * n_u
end

function test_calculate_const_initial_penalty_linearised()
    print_message("calculate_const_initial_penalty_linearised")
    n_x, n_u, N = 4, 2, 11
    scvx = SCvx(N=N, n_x=n_x, n_u=n_u,
                consts_initial_eq=[my_const_initial_eq],
               )
    X, U = scvx.X_k, scvx.U_k  # zeros
    D, W = X, U  # originally, it should be values from Convex.Variable
    jacob_dict = Dict()
    obj, _ = SCvxs.calculate_const_initial_penalty_linearised(scvx, D, W, jacob_dict)
    @test obj == 0.0
end

function test_calculate_const_terminal_penalty_linearised()
    print_message("calculate_const_terminal_penalty_linearised")
    n_x, n_u, N = 4, 2, 11
    scvx = SCvx(N=N, n_x=n_x, n_u=n_u,
                consts_terminal_eq=[my_const_terminal_eq]
               )
    X, U = scvx.X_k, scvx.U_k  # zeros
    D, W = X, U  # originally, it should be values from Convex.Variable
    jacob_dict = Dict()
    obj, _ = SCvxs.calculate_const_terminal_penalty_linearised(scvx, D, W, jacob_dict)
    @test obj == 2.0 * n_x * scvx.λ
end

################## get_L ##################

function test_solve_cvx_subprob()
    print_message("solve_cvx_subprob")
    n_x, n_u, N = 4, 2, 11
    scvx = SCvx(N=N, n_x=n_x, n_u=n_u,
                objs_path=[my_path_obj],
                objs_terminal=[my_terminal_obj],
                consts_path_ineq=[not_too_large_input],
                consts_path_eq=[my_const_path_eq],
                consts_initial_eq=[my_const_initial_eq],
                consts_terminal_eq=[my_const_terminal_eq],
               )
    D, W, jacob_dict = SCvxs.solve_cvx_subprob(scvx)
    println("D: $(D.value)")
    println("W: $(W.value)")
end

function test_scvx_example(; verbose=false)
    n_x, n_u, N = 4, 2, 31
    scvx = SCvx(N=N, n_x=n_x, n_u=n_u,
                objs_path=[my_path_obj],
                objs_terminal=[my_terminal_obj],
                consts_path_ineq=[not_too_large_input2,
                                  not_too_small_input],
                consts_path_eq=[my_const_path_eq],
                consts_initial_eq=[my_const_initial_eq],
                consts_terminal_eq=[my_const_terminal_eq],
               )
    X_0, U_0 = ones(N, n_x), ones(N-1, n_u)
    scvx = initial_guess!(scvx, X_0, U_0)
    @time solve!(scvx, verbose=verbose)
end

function test_solve(;verbose=false)
    print_message("solve")
    scvx = test_scvx_example(verbose=false)
    N = scvx.N
    @show scvx.obj_k, scvx.i, scvx.flag
    @show scvx.X_k
    @show scvx.U_k
    # initial condition
    tol = 1e-4
    @test norm(my_const_initial_eq(scvx.X_k[1, :], scvx.U_k[1, :])) < tol
    @test norm(my_const_terminal_eq(scvx.X_k[end, :])) < tol
    for n in 1:N-1
        @test all(not_too_large_input2(scvx.X_k[n, :],
                                       scvx.U_k[n, :]) .<= tol)
        @test norm(my_const_path_eq(scvx.X_k[n, :],
                                    scvx.U_k[n, :],
                                    scvx.X_k[n+1, :])) < tol
    end
    # plot
    data = Dict("time_state" => collect(1:N),
                "time_input" => collect(1:N-1),
                "state" => scvx.X_k, "input" => scvx.U_k)
    @show data["state"]
    @show data["input"]
    _plot(data)
end

function _plot(data)
    log_dir = "data/test"
    mkpath(log_dir)
    for name in ["state", "input"]
        time = data["time_" * name]
        p = plot(time, data[name], lw=3)
        if name == "state"
            plot!(seriestype=:scatter,
                  [time[1]],
                  [zeros(size(data[name][1, :]))],
                  markercolor="red",
                  label="init cond",
                 )  # init cond
            plot!(seriestype=:scatter,
                  [time[end]],
                  [2*ones(size(data[name][end, :]))],
                  markercolor="red",
                  label="term cond",
                 )  # term cond
        elseif name == "input"
            plot!(time, 2*ones(size(time)),  # not too large
                  linecolor="red",
                  label="not too large",
                 )
            plot!(time, -2*ones(size(time)),  # not too small
                  linecolor="red",
                  label="not too small",
                 )
        end
        savefig(p, joinpath(log_dir, name*".pdf"))
    end
end

################## custom functions ##################
function my_path_obj(x::Array, u::Array)::Array
    return [0.5*(norm(x)^2 + norm(u)^2)]
end

function my_terminal_obj(x::Array)::Array
    return [norm(x .- 2.0)^2]
end

function not_too_large_input(x::Array, u::Array)::Array
    return u .+ 2.0  # <=0
end

function not_too_large_input2(x::Array, u::Array)::Array
    return u .- 2.0  # <=0
end

function not_too_small_input(x::Array, u::Array)::Array
    return -(u .+ 2.0)  # <=0
end

function my_const_path_eq(x::Array, u::Array, x_next::Array)::Array
    A = Matrix(I, 4, 4)
    B = [0 1; 0 1; 1 0; 1 0]
    dynamics(x, u) = A*x + B*u
    return x_next - dynamics(x, u)
end


function my_const_initial_eq(x::Array, u::Array)
    return x
end

function my_const_terminal_eq(x::Array)
    return (x .- 2.0)
end

################## test all ##################
function test_all()
    # test_initial_guess()

    # ################## get_J ##################
    # test_get_J()
    # ## get_obj
    # test_get_obj()
    # test_calculate_objs_path()
    # test_calculate_objs_terminal()

    # ## get_obj_const_penalty
    # test_get_obj_const_penalty()
    # test_calculate_const_path_penalty()
    # test_calculate_const_initial_penalty()
    # test_calculate_const_terminal_penalty()

    # ################## get_L ##################
    # test_get_L()
    # ## get_obj_linearised
    # test_get_obj_linearised()
    # test_calculate_objs_path_linearised()
    # test_calculate_objs_terminal_linearised()
    # ## get_obj_const_penalty_linearised
    # test_get_obj_const_penalty_linearised()
    # test_calculate_const_path_penalty_linearised()
    # test_calculate_const_initial_penalty_linearised()
    # test_calculate_const_terminal_penalty_linearised()

    # ################## solve ##################
    # test_solve_cvx_subprob()
    test_solve(verbose=false)
    # ################## solve ##################
    # test_scvx_example()
end

test_all()
