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
