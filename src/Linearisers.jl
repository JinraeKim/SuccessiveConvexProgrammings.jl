module Linearisers

using ForwardDiff

export get_jacobian


"""
    get_jacobian(func, args...)

Compute the jacobian of given function `func`.
`args...` should be arguments of `func`.

# Equations
jacobian = [[∂f1/∂x1 ... ∂f1/∂xn];
                ⋮           ⋮
            [∂fm/∂x1 ... ∂fm/∂xn]]

# Examples
```julia
using SuccessiveConvexProgrammings
using LinearAlgebra


function my_func_array(x, u, t, k)
    A = [1 2;
         3 4;
         5 6]
    B = Matrix(I, 3, 3)
    return A*x + B*u
end

x = [1, 2]
u = [3, 4, 5]
t = [6, 7]
k = [8, 9]
jacob = get_jacobian(my_func_array, x, u, t, k)
# result
3×9 Array{Float64,2}:
 1.0  2.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0
 3.0  4.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0
 5.0  6.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0
```
"""
function get_jacobian(func, args...)
    jacob = zeros(length(func(args...)), 0)  # dummy calculation
    args_dim = 0
    for (i, arg) in enumerate(args)
        args_tmp = [args...]
        args_tmp[1], args_tmp[i] = args_tmp[i], args_tmp[1]
        altered_func = function(args2...)
            args_tmp2 = [args2...]
            args_tmp2[1], args_tmp2[i] = args_tmp2[i], args_tmp2[1]
            return func(args_tmp2...)
        end
        new_func = function(arg)
            return altered_func(arg, args_tmp[2:end]...)
        end
        jacob = cat(jacob, ForwardDiff.jacobian(new_func, arg), dims=2)
    end
    return jacob
end


end
