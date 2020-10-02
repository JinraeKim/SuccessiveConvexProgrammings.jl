# SuccessiveConvexProgrammings
**SuccessiveConvexProgrammings** (a.k.a. SCPs)
is for Successive Convex Programming algorithms.
Currently,
this repo focuses on the realisation of the existing algorithms.

## Usage
Here's an example of the usage of
[Y. Mao et al., Successive Convexification, 2018](https://arxiv.org/abs/1804.06539),
one of SCP algorithms for solving non-convex optimal control problem.

```julia
# Custom functions for problem formulation
function my_path_obj(x::Array, u::Array)::Array
    return [0.5*(norm(x)^2 + norm(u)^2)]
end

function my_terminal_obj(x::Array)::Array
    return [norm(x .- 2.0)^2]
end

function not_too_large_input(x::Array, u::Array)::Array  # not used here
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


# SCvx example
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
```

Result:

```julia
  0.921291 seconds (4.62 M allocations: 299.719 MiB, 5.15% gc time)
SCvx
  N: Int64 31
  i: Int64 14
  i_max: Int64 200
  n_x: Int64 4
  n_u: Int64 2
  r_k: Float64 0.5
  λ: Float64 100000.0
  ρ0: Float64 0.0
  ρ1: Float64 0.25
  ρ2: Float64 0.7
  rl: Float64 0.1
  α: Float64 2.0
  β: Float64 2.0
  ϵ: Float64 0.001
  objs_path: Array{typeof(my_path_obj)}((1,))
  objs_terminal: Array{typeof(my_terminal_obj)}((1,))
  consts_path_ineq: Array{Function}((2,))
  consts_path_eq: Array{typeof(my_const_path_eq)}((1,))
  consts_initial_ineq: Nothing nothing
  consts_initial_eq: Array{typeof(my_const_initial_eq)}((1,))
  consts_terminal_ineq: Nothing nothing
  consts_terminal_eq: Array{typeof(my_const_terminal_eq)}((1,))
  X_k: Array{Float64}((31, 4)) [-1.3511982930901345e-15 -1.3548666104268918e-15 -1.380711075409979e-15 -1.3816195819191138e-15; -0.00022117647021521698 -0.00022117647021522433 -0.0002200495650853284 -0.0002200495650853302; … ; 0.5772280007439645 0.5772280007439643 0.5772237620536798 0.5772237620536798; 2.000000000537185 2.000000000537185 2.000000000537199 2.000000000537199]
  U_k: Array{Float64}((30, 2)) [-0.00022004956515178924 -0.00022117647028205013; -6.284246324169491e-5 -6.288074341830041e-5; … ; 0.41606753466786084 0.4160697952736614; 1.4227762395362158 1.4227720008458915]
  obj_k: Float64 2.9282206598904503
  flag: String "success"
```

See directory `test` for more details.
