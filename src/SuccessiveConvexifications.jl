module SuccessiveConvexifications
"""
module SuccessiveConvexifications a.k.a. SCvx

[Ref] Mao, Y., Szmuk, M., Xu, X., & Acikmese, B. (2018).
Successive Convexification:
A Superlinearly Convergent Algorithm for Non-convex Optimal Control
Problems.

[Note]
N > 0: discrete-time horizon length
x ∈ R^(n_x, 1): state
u ∈ R^(n_u, 1): input
X ∈ R^(N, n_x): concatenated state, X = [x1 x2 ... xN]
U ∈ R^((N-1), n_u): concatenated input, U = [u1 u2 ... u(N-1)]
D ∈ R^(N, n_x): optimisation variables of convex subproblem
w.r.t state, D = [d1 d2 ... dN]
W ∈ R^((N-1), n_u): optimisation variables of convex subproblem
w.r.t input, W = [w1 w2 ... w(N-1)]
"""

using Parameters
using Convex
const Cvx = Convex  # to clarify the code
using Mosek
using MosekTools
const SOLVER = Mosek
# using Debugger

using Reexport
include("Linearisers.jl")
@reexport using .Linearisers


export SCvx
export initial_guess!, solve!, flush!

##################### Extension ####################
function Convex.pos(x::Array)
    y = deepcopy(x)
    for i in 1:length(y)
        if y[i] < 0
            y[i] = 0.0
        end
    end
    return y
end

##################### SuccessiveConvexifications ####################
"""
# Note
`objs` and `consts` is an array containing obj. funcs. and constraint funcs.,
resp.
1) The objective function is the sum of obj. funcs. in `objs` arrays.
2) The constraint function is combination of all const. funcs. in `consts` arrays.
    1) Inequality constraint: f ≤ 0
    2) Equality constraint: f = 0
# Arguments
- `objs_(option)::Array`: contains objective functions.
    - option: `path` or `terminal`
- `consts_(option1)_(option2)::Array`: contains constraint functions.
    - option1: `path`, `initial` or `terminal
    - option2: `eq` or `ineq`
"""
@with_kw mutable struct SCvx
    # hyperparameters
    N::Int64;  @assert N > 2
    i::Int64 = 0
    i_max::Int64 = 200
    n_x::Int64
    n_u::Int64
    r_k::Float64 = 1e0
    λ::Float64 = 1e5
    ρ0::Float64 = 0.0
    ρ1::Float64 = 0.25
    ρ2::Float64 = 0.7
    rl::Float64 = 1.0e-1
    α::Float64 = 2.0e0
    β::Float64 = 2.0e0
    ϵ::Float64 = 1e-3
    # problem
    objs_path::Array = []
    objs_terminal::Array = []
    consts_path_ineq::Array = []
    consts_path_eq::Array = []
    consts_initial_ineq::Array = []
    consts_initial_eq::Array = []
    consts_terminal_ineq::Array = []
    consts_terminal_eq::Array = []
    # default initial guess
    X_k::Array = zeros(N, n_x)
    U_k::Array = zeros(N-1, n_u)
    obj_k::Number = -Inf  # indicates that it does not solved yet
    flag::String = ""
end

function initial_guess!(scvx::SCvx, X_k::Array, U_k::Array)
    if scvx.N != size(X_k)[1] || scvx.N != (size(U_k)[1]+1)
        error("Invalid length")
    elseif scvx.n_x != size(X_k)[2] || scvx.n_u != size(U_k)[2]
        error("Dimension disagreement")
    end
    scvx.X_k, scvx.U_k = X_k, U_k
    return scvx
end

"""
    flush!(scvx)

Flush the initial values of solution.
You would need this when applying this algorithm to fixed final time problem.
"""
function flush!(scvx::SCvx)
    scvx.N = scvx.N - 1
    scvx.X_k = scvx.X_k[2:end, :]
    scvx.U_k = scvx.U_k[2:end, :]
end

function Cvx.solve!(scvx::SCvx; verbose::Bool=false)
    is_not_skipped = true
    is_not_stopped = true
    scvx.i = 0  # reset
    while is_not_stopped
        # step 1: solve convex subproblem
        D, W, jacob_dict = solve_cvx_subprob(scvx, verbose=verbose)
        # step 2: compute the actual and predicted change
        J_k, scvx.obj_k = get_J(scvx, scvx.X_k, scvx.U_k, with_obj=true)
        diff_J = J_k - get_J(scvx, scvx.X_k+D.value, scvx.U_k+W.value)
        diff_L = J_k - get_L(scvx, D.value, W.value, jacob_dict)
        # if diff_J == 0.0  # TODO: should be diff_L == 0.0 ?
        if diff_L == 0.0
            scvx.flag = "diff_L = 0"
            break
        else
            ρ_k = diff_J / diff_L
        end
        # step 3: compare the ratio
        if ρ_k <= scvx.ρ0
            scvx.r_k = scvx.r_k /scvx.α  # reject and contracted
            is_not_skipped = false  # go to step 1
            scvx.flag = "rejected"
        else
            is_not_skipped = true
        end
        if is_not_skipped
            scvx.X_k, scvx.U_k = (
                                  scvx.X_k + D.value, scvx.U_k + W.value
                                 )  # accept
            if ρ_k < scvx.ρ1
                r_k = scvx.r_k / scvx.α
                scvx.flag = "accept but contracted"
            elseif ρ_k < scvx.ρ2
                r_k = scvx.r_k
                scvx.flag = "accept and maintain"
            else
                r_k = scvx.r_k * scvx.β
                scvx.flag = "accept and enlarged"
            end
            scvx.r_k = max(r_k, scvx.rl)
            is_not_stopped = diff_J > scvx.ϵ && scvx.i < scvx.i_max
            if !is_not_stopped
                if scvx.i >= scvx.i_max
                    scvx.flag = "maximum iteration"
                elseif diff_J <= scvx.ϵ
                    scvx.flag = "success"
                end
            end
        end
        scvx.i += 1
        if verbose && scvx.i % 10 == 0
            println("iter = $(scvx.i)")
        end
    end
    return scvx
end

function solve_cvx_subprob(scvx::SCvx; verbose::Bool=false)
    N = scvx.N
    n_x, n_u = scvx.n_x, scvx.n_u
    D, W = Cvx.Variable(N, n_x), Cvx.Variable(N-1, n_u)
    ## problem formulation
    # constraints
    constraints = [norm([reshape(D, prod(size(D)), 1);
                         reshape(W, prod(size(W)), 1)], 2) <= scvx.r_k]
    # linearised and penalised objective
    obj_linearised, jacob_dict = get_obj_linearised(scvx,
                                                    D, W,
                                                    jacob_dict=Dict())
    (
     obj_const_penalty_linearised,
     jacob_dict
    ) = get_obj_const_penalty_linearised(scvx,
                                         D, W, jacob_dict=jacob_dict)
    # TODO: add convex objective and constraints for path and terminal
    obj_linearised_penalised = (obj_linearised
                                + obj_const_penalty_linearised)
    prob = minimize(obj_linearised_penalised, constraints)
    if SOLVER == Mosek
        Cvx.solve!(prob, SOLVER.Optimizer(LOG=verbose))  # for Mosek
    else
        Cvx.solve!(prob, SOLVER.Optimizer(verbose=verbose))  # for others
    end
    return D, W, jacob_dict
end

################## get_J ##################
function get_J(scvx::SCvx, X, U; with_obj=false)
    obj = get_obj(scvx, X, U)
    const_penalty = get_obj_const_penalty(scvx, X, U)
    J = obj + const_penalty
    if with_obj
        return J, obj
    else
        return J
    end
end

## get_obj
function get_obj(scvx::SCvx, X, U)
    obj = 0.0
    obj += calculate_objs_path(scvx, X, U)
    obj += calculate_objs_terminal(scvx, X)
    return obj
end

function calculate_objs_path(scvx::SCvx, X, U)::Number
    # path obj
    N = scvx.N
    obj = zeros(1)
    if scvx.objs_path == []
    else
        for n in 1:N-1
            for obj_path in scvx.objs_path
                X_n, U_n = X[n, :], U[n, :]
                obj += obj_path(X_n, U_n)
            end
        end
    end
    return obj[1]
end

function calculate_objs_terminal(scvx::SCvx, X)::Number
    # terminal obj
    obj = zeros(1)
    if scvx.objs_terminal == []
    else
        for obj_terminal in scvx.objs_terminal
            X_N = X[end, :]
            obj += obj_terminal(X_N)
        end
    end
    return obj[1]
end

## get_obj_const_penalty
function get_obj_const_penalty(scvx::SCvx, X, U)
    obj = 0.0
    obj += calculate_const_path_penalty(scvx, X, U)
    obj += calculate_const_initial_penalty(scvx, X, U)
    obj += calculate_const_terminal_penalty(scvx, X)
    return obj
end

function calculate_const_path_penalty(scvx::SCvx, X, U)
    N = scvx.N
    obj = 0.0
    for n in 1:N-1
        X_n, U_n, X_n_next = X[n, :], U[n, :], X[n+1, :]
        if scvx.consts_path_ineq == []
        else
            for const_ineq in scvx.consts_path_ineq
                obj += calculate_ineq_penalty(scvx, const_ineq, X_n, U_n)
            end
        end
        if scvx.consts_path_eq == []
        else
            for const_eq in scvx.consts_path_eq
                obj += calculate_eq_penalty(scvx,
                                            const_eq, X_n, U_n, X_n_next)
            end
        end
    end
    return obj
end

function calculate_const_initial_penalty(scvx::SCvx, X, U)
    X_1, U_1 = X[1, :], U[1, :]
    obj = 0.0
    if scvx.consts_initial_ineq == []
    else
        for const_ineq in scvx.consts_initial_ineq
            obj += calculate_ineq_penalty(scvx, const_ineq, X_1, U_1)
        end
    end
    if scvx.consts_initial_eq == []
    else
        for const_eq in scvx.consts_initial_eq
            obj += calculate_eq_penalty(scvx, const_eq, X_1, U_1)
        end
    end
    return obj
end

function calculate_const_terminal_penalty(scvx::SCvx, X)
    X_N = X[end, :]
    obj = 0.0
    if scvx.consts_terminal_ineq == []
    else
        for const_ineq in scvx.consts_terminal_ineq
            obj += calculate_ineq_penalty(scvx, const_ineq, X_N)
        end
    end
    if scvx.consts_terminal_eq == []
    else
        for const_eq in scvx.consts_terminal_eq
            obj += calculate_eq_penalty(scvx, const_eq, X_N)
        end
    end
    return obj
end

################## get_L ##################
function get_L(scvx::SCvx, D::Array, W::Array, jacob_dict)::Number
    obj_linearised, _ = get_obj_linearised(scvx,
                                           D, W,
                                           jacob_dict=jacob_dict)
    (
     obj_const_penalty_linearised, _
    ) = get_obj_const_penalty_linearised(scvx,
                                         D, W, jacob_dict=jacob_dict)
    L = obj_linearised + obj_const_penalty_linearised
    return L
end

## get_obj_linearised
function get_obj_linearised(scvx::SCvx, D, W; jacob_dict=Dict())
    obj = 0.0
    obj_path, jacob_dict = calculate_objs_path_linearised(scvx,
                                                          D, W,
                                                          jacob_dict)
    obj += obj_path  # path
    (
     obj_terminal, jacob_dict
    ) = calculate_objs_terminal_linearised(scvx, D, W, jacob_dict)
    obj += obj_terminal  # terminal
    return obj, jacob_dict
end

function calculate_objs_path_linearised(scvx::SCvx, D, W, jacob_dict)
    name = "objs_path"
    if !haskey(jacob_dict, name)
        jacob_dict[name] = []
    end
    N = scvx.N
    X_k, U_k = scvx.X_k, scvx.U_k  # linearisation point
    obj = 0.0
    DW = [D[1:end-1, :] W]
    if scvx.objs_path == []
    else
        for n in 1:N-1
            X_k_n, U_k_n = X_k[n, :], U_k[n, :]
            for obj_path in scvx.objs_path
                DW_n = DW[n, :]
                (
                 jacob, jacob_dict
                ) = use_or_calculate_jacobian(name, size(jacob_dict[name])[1] < N-1, jacob_dict, obj_path, X_k_n, U_k_n)
                obj_path_tmp = (obj_path(X_k_n, U_k_n)
                                + jacob * reshape(DW_n,
                                                  prod(size(DW_n)), 1)
                               )[1]  # reshape for Cvx.Variable
                obj += obj_path_tmp
            end
        end
    end
    return obj, jacob_dict
end

function calculate_objs_terminal_linearised(scvx::SCvx, D, W, jacob_dict)
    name = "objs_terminal"
    if !haskey(jacob_dict, name)
        jacob_dict[name] = []
    end
    obj = 0.0
    X_k = scvx.X_k
    if scvx.objs_terminal == []
    else
        for obj_terminal in scvx.objs_terminal
            X_k_N = X_k[end, :]
            D_N = D[end, :]
            name = "objs_terminal"
            (
             jacob, jacob_dict
            ) = use_or_calculate_jacobian(name, size(jacob_dict[name])[1] == 0, jacob_dict, obj_terminal, X_k_N)
            obj_tmp = (obj_terminal(X_k_N)
                       + jacob * reshape(D_N, prod(size(D_N)), 1)
                      )[1]  # reshape for Cvx.Variable
            obj += obj_tmp
        end
    end
    return obj, jacob_dict
end

## get_obj_const_penalty_linearised
function get_obj_const_penalty_linearised(scvx::SCvx, D, W;
                                          jacob_dict=Dict())
    obj = 0.0
    (
     obj_path, jacob_dict
    ) = calculate_const_path_penalty_linearised(scvx, D, W, jacob_dict)
    obj += obj_path  # path
    (
     obj_initial, jacob_dict
    ) = calculate_const_initial_penalty_linearised(scvx, D, W, jacob_dict)
    obj += obj_initial  # initial
    (
     obj_terminal, jacob_dict
    ) = calculate_const_terminal_penalty_linearised(scvx,
                                                    D, W, jacob_dict)
    obj += obj_terminal  # terminal
    return obj, jacob_dict
end

function calculate_const_path_penalty_linearised(scvx::SCvx,
                                                 D, W, jacob_dict)
    name_ineq = "consts_path_ineq"
    if !haskey(jacob_dict, name_ineq)
        jacob_dict[name_ineq] = []
    end
    name_eq = "consts_path_eq"
    if !haskey(jacob_dict, name_eq)
        jacob_dict[name_eq] = []
    end
    obj = 0.0
    N = scvx.N
    X_k, U_k = scvx.X_k, scvx.U_k  # linearisation point
    DW = [D[1:end-1, :] W]
    DW_extended = [D[1:end-1, :] W D[2:end, :]]

    for n in 1:N-1
        X_k_n, U_k_n, X_k_n_next = X_k[n, :], U_k[n, :], X_k[n+1, :]
        DW_n = DW[n, :]
        D_n, W_n, D_n_next = D[n, :], W[n, :], D[n+1, :]
        DW_n_extended = DW_extended[n, :]
        if scvx.consts_path_ineq == []
        else
            for const_ineq in scvx.consts_path_ineq
                (
                 jacob, jacob_dict
                ) = use_or_calculate_jacobian(name_ineq, size(jacob_dict[name_ineq])[1] < N-1, jacob_dict, const_ineq, X_k_n, U_k_n, n=n)
                const_ineq_tmp(
                               X_k_n, U_k_n, DW_n, jacob
                              ) = (
                                   const_ineq(X_k_n, U_k_n)
                                   + jacob * reshape(DW_n,
                                                     prod(size(DW_n)), 1)
                                  )  # reshape for Cvx.Variable
                obj += calculate_ineq_penalty(scvx, const_ineq_tmp,
                                              X_k_n, U_k_n, DW_n, jacob)
            end
        end
        if scvx.consts_path_eq == []
        else
            for const_eq in scvx.consts_path_eq
                (
                 jacob, jacob_dict
                ) = use_or_calculate_jacobian(name_eq, size(jacob_dict[name_ineq])[1] < N-1, jacob_dict, const_eq, X_k_n, U_k_n, X_k_n_next, n=n)
                const_eq_tmp(
                             X_k_n, U_k_n, X_k_n_next,
                             DW_n_extended, jacob
                            ) = (
                                 const_eq(X_k_n, U_k_n, X_k_n_next)
                                 + jacob * reshape(DW_n_extended, prod(size(DW_n_extended)), 1)
                                )  # reshape for Cvx.Variable
                obj += calculate_eq_penalty(scvx, const_eq_tmp,
                                            X_k_n, U_k_n, X_k_n_next,
                                            DW_n_extended, jacob)
            end
        end
    end
    return obj, jacob_dict
end

function calculate_const_initial_penalty_linearised(scvx::SCvx,
                                                    D, W, jacob_dict)
    name_ineq = "consts_initial_ineq"
    if !haskey(jacob_dict, name_ineq)
        jacob_dict[name_ineq] = []
    end
    name_eq = "consts_initial_eq"
    if !haskey(jacob_dict, name_eq)
        jacob_dict[name_eq] = []
    end
    obj = 0.0
    X_k, U_k = scvx.X_k, scvx.U_k  # linearisation point
    X_k_1, U_k_1 = X_k[1, :], U_k[1, :]
    DW = [D[1:end-1, :] W]
    DW_1 = DW[1, :]
    D_1, W_1 = D[1, :], W[1, :]
    if scvx.consts_initial_ineq == []
    else
        for const_ineq in scvx.consts_initial_ineq
            (
             jacob, jacob_dict
            ) = use_or_calculate_jacobian(name_ineq, size(jacob_dict[name_ineq])[1] == 0, jacob_dict, const_ineq, X_k_1, U_k_1)
            const_ineq_tmp(
                           X_k_1, U_k_1, DW_1, jacob
                          ) = const_ineq(X_k_1, U_k_1) + jacob * DW_1
            obj += calculate_ineq_penalty(scvx, const_ineq_tmp,
                                          X_k_1, U_k_1, DW_1, jacob)
        end
    end
    if scvx.consts_initial_eq == []
    else
        for const_eq in scvx.consts_initial_eq
            (
             jacob, jacob_dict
            ) = use_or_calculate_jacobian(name_eq, size(jacob_dict[name_eq])[1] == 0, jacob_dict, const_eq, X_k_1, U_k_1)
            const_eq_tmp(
                         X_k_1, U_k_1, DW_1, jacob
                        ) = (
                             const_eq(X_k_1, U_k_1)
                             + jacob * reshape(DW_1, prod(size(DW_1)), 1)
                            )  # reshape for Cvx.Variable
            obj += calculate_eq_penalty(scvx, const_eq_tmp,
                                        X_k_1, U_k_1, DW_1, jacob)
        end
    end
    return obj, jacob_dict
end

function calculate_const_terminal_penalty_linearised(scvx::SCvx,
                                                     D, W, jacob_dict)
    name_ineq = "consts_terminal_ineq"
    if !haskey(jacob_dict, name_ineq)
        jacob_dict[name_ineq] = []
    end
    name_eq = "consts_terminal_eq"
    if !haskey(jacob_dict, name_eq)
        jacob_dict[name_eq] = []
    end
    obj = 0.0
    X_k, U_k = scvx.X_k, scvx.U_k  # linearisation point
    X_k_N = X_k[end, :]
    D_N = D[end, :]
    if scvx.consts_terminal_ineq == []
    else
        for const_ineq in scvx.consts_terminal_ineq
            (
             jacob, jacob_dict
            ) = use_or_calculate_jacobian(name_ineq, size(jacob_dict[name_ineq])[1] == 0, jacob_dict, const_ineq, X_k_N)
            const_ineq_tmp(
                           X_k_N, D_N, jacob
                          ) = (
                               const_ineq(X_k_N)
                               + jacob * reshape(D_N, prod(size(D_N)), 1)
                              )  # reshape for Cvx.Variable
            obj += calculate_ineq_penalty(scvx, const_ineq_tmp,
                                          X_k_N, D_N, jacob)
        end
    end
    if scvx.consts_terminal_eq == []
    else
        for const_eq in scvx.consts_terminal_eq
            (
             jacob, jacob_dict
            ) = use_or_calculate_jacobian(name_eq, size(jacob_dict[name_eq])[1] == 0, jacob_dict, const_eq, X_k_N)
            const_eq_tmp(
                         X_k_N, D_N, jacob
                        ) = (
                             const_eq(X_k_N)
                             + jacob * reshape(D_N, prod(size(D_N)), 1)
                            )  # reshape for Cvx.Variable
            obj += calculate_eq_penalty(scvx, const_eq_tmp,
                                        X_k_N, D_N, jacob)
        end
    end
    return obj, jacob_dict
end

################## etc ##################
function calculate_ineq_penalty(scvx::SCvx, func, args...)
    return scvx.λ * sum(pos(func(args...)))
end

function calculate_eq_penalty(scvx::SCvx, func, args...)
    return scvx.λ * norm(func(args...), 1)
end

function use_or_calculate_jacobian(name, condition, jacob_dict, func,
                                   args...; n=1)
    # TODO: add convex check
    if condition
        jacob = get_jacobian(func, args...)
        jacob_dict[name] = [jacob]
    else
        jacob = jacob_dict[name][n]
    end
    return jacob, jacob_dict
end


end
