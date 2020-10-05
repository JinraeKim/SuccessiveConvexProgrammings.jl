module Discretisers

using FymEnvs
using LinearAlgebra
using Debugger
include("Linearisers.jl")
using .Linearisers: get_jacobian

export Discretiser, ZeroOrderHoldDiscretiser
export discretise


abstract type Discretiser end

mutable struct ZeroOrderHoldDiscretiser <: Discretiser
    env
    nonlinear_dynamics  # f(x, u)
    x_k  # function of t
    u_k  # function of t
    ZeroOrderHoldDiscretiser(args...; kwargs...) = init!(new(), args...; kwargs...)
end

function init!(disc::ZeroOrderHoldDiscretiser, nonlinear_dynamics;
               ts=nothing, x_k=nothing, u_k=nothing, kwargs...)
    dt = ts[2] - ts[1]
    if -dt >= 0.0
        error("Invalid order of time")
    elseif ts == nothing || x_k == nothing || u_k == nothing
        error("No linearisation points")
    elseif length(size(x_k)) > 2 || length(size(u_k)) > 2
        error("Only one dimensional state and input are available")
    end
    n, m = size(x_k)[2], size(u_k)[2]
    disc.nonlinear_dynamics = nonlinear_dynamics
    disc.x_k = time_indexing(x_k, ts, dt)
    disc.u_k = time_indexing(u_k, ts, dt)

    function set_dyn(env, t)  # values at t
        # x_k, u_k: linearisation point
        Ad = env.systems["Ad"]
        Bd = env.systems["Bd"]
        cd = env.systems["cd"]
        jacob = get_jacobian(disc.nonlinear_dynamics, disc.x_k(t), disc.u_k(t))
        A = jacob[:, 1:n]
        B = jacob[:, n+1:n+m]
        c = disc.nonlinear_dynamics(disc.x_k(t), disc.u_k(t)) - A*disc.x_k(t) - B*disc.u_k(t)
        Ad.dot = - Ad.state * A
        Bd.dot = - Ad.state * B
        cd.dot = - Ad.state * c
    end
    function step(env)
        update!(env)
        # after update
        Ad = env.systems["Ad"]
        Bd = env.systems["Bd"]
        cd = env.systems["cd"]
        info = Dict(
                    "Ad" => Ad.state,
                    "Bd" => Bd.state,
                    "cd" => cd.state,
                   )
        done = time_over(env.clock)
        return done, info
    end
    disc.env = BaseEnv(;initial_time=ts[end], dt=-dt, max_t=ts[1], kwargs...)
    env = disc.env
    systems = Dict(
                   "Ad" => BaseSystem(initial_state=Matrix(I, n, n), name="Ad"),
                   "Bd" => BaseSystem(initial_state=zeros(n, m), name="Bd"),
                   "cd" => BaseSystem(initial_state=zeros(n), name="cd"),
                  )  # initial state: values at t_{i+1}
    systems!(env, systems)
    dyn!(env, set_dyn)
    step!(env, step)
    return disc
end

function time_indexing(xs, ts, dt)
    if size(xs)[1] != length(ts)
        error("Length mismatched")
    elseif dt < 0.0
        error("Invalid time step for time indexing")
    end
    func = function(t)
        idx = findfirst(x -> abs(x-t) < dt , ts)
        return xs[idx, :]
    end
    return func
end

"""
    zero_order_hold

# Equations
x_{i+1} = Ad x_i + Bd u_i + cd
where
Ad = Φ(t_{i+1}, t_i)
Bd = ∫_{t_i}^{t_{i+1}} Φ(t_{i+1}, τ) B(τ) dτ
cd = ∫_{t_i}^{t_{i+1}} Φ(t_{i+1}, τ) c(τ) dτ
"""
function discretise(disc::ZeroOrderHoldDiscretiser)
    env = disc.env
    reset!(env)

    Ad = []
    Bd = []
    cd = []
    while true
        reinit!(disc)  # need to be reinitialised for every step
        done, info = env.step()  # backward integration
        push!(Ad, info["Ad"])
        push!(Bd, info["Bd"])
        push!(cd, info["cd"])
        if done
            reverse!(Ad)
            reverse!(Bd)
            reverse!(cd)
            break
        end
    end
    return Ad, Bd, cd
end

"""
    reinit!(disc)

Re-initialise the state of disc's systems while preserving time.
"""
function reinit!(disc::ZeroOrderHoldDiscretiser)
    env = disc.env
    for name in ["Ad", "Bd", "cd"]
        env.systems[name].state = env.systems[name].initial_state
    end
end


end  # module
