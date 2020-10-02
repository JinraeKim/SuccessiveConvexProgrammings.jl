using LinearAlgebra
using ForwardDiff
using SuccessiveConvexProgrammings

# test
function my_func(x, u, t, k)
    A = [1 2; 3 4]
    return A*x
end

function my_func_array(x, u, t, k)
    A = [1 2; 3 4; 5 6]
    B = Matrix(I, 3, 3)
    # @bp
    return A*x + B*u
end

x = [1; 2]
u = [3; 4; 5]
t = [5; 6]
k = [7; 8]
@run jacob = get_jacobian(my_func_array, x, u, t, k)
