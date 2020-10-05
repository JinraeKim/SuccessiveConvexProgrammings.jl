using SuccessiveConvexProgrammings
using Test

using FymEnvs  # TODO: delete it
using LinearAlgebra


function print_message(title)
    println(">"^6 * title * "<"^6)
end

function my_dynamics(x, u)
    A = [0 1 0; 0 0 1; 0 0 0]
    return A*x + sin.(u)
end


function test_zero_order_hold()
    print_message("zero_order_hold")
    dt = 0.01
    ts = collect(0:dt:0.1)
    # linearisation points
    x_k = zeros(length(ts), 3)
    u_k = (pi/4) * ones(length(ts), 3)

    disc = ZeroOrderHoldDiscretiser(my_dynamics,
                                    ts=ts, x_k=x_k, u_k=u_k)
    Ad, Bd, cd = discretise(disc)

    eps = 1e-4
    for i in 1:length(ts)
        @test norm(Ad[i] - exp(A * dt)) < eps
    end
end

test_zero_order_hold()
