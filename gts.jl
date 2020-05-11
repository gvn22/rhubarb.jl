using DifferentialEquations, LinearAlgebra
using StaticArrays, SparseArrays, BenchmarkTools
using Plots

# function lorenz!(du,u,p,t)
#     # Lorenz Equations
#     x,y,z = u
#     σ,ρ,β = p
#
#     du[1] = dx = σ*(y-x)
#     du[2] = dy = x*(ρ-z) - y
#     du[3] = dz = x*y - β*z
# end

function lv!(du,u,p,t)
    # Lotka-Volterra Equations
    # These have a similar structure as GQL equations
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α*x - β*x*y
    du[2] = dy = -δ*y + γ*x*y
end

function L(p)

    α, δ = p
    return [α 0.0;
            0.0 -δ]

end

function N(p)

    β, γ = p
    return [0.0 -β;
             γ 0.0]

end

function F(p)

    τ = p
    return (0.0/τ) .* [1.0 0.0;
                       0.0 1.0]

end

function glv!(du,u,p,t)

    # Generalise LV towards GQL
    α, β, δ, γ, τ = p
    # a = 0.0
    b = L([α,δ]) * u
    c = N([β,γ]) .* (u*u')
    # d = F([τ]) * u

    for i ∈ [1,2]
        du[i] = b[i] + sum(c[i,:])
    end

end

u0 = [1.0,1.0]
tspan = (0.0,30.0)
p = [1.5,1.0,3.0,1.0,1.0]
prob = ODEProblem(glv!,u0,tspan,p)
sol  = solve(prob,RK4(),adaptive=true)
plot(sol,vars=(1))
plot!(sol,vars=(2))

# u0 = [1.0;0.0;0.0]
# tspan = (0.0,100.0)
# p = [10.0,28.0,8/3]
#
# prob = ODEProblem(lorenz!,u0,tspan,p)
# sol  = solve(prob,RK4(),adaptive=true)
# @benchmark sol = solve(prob,Tsit5(),save_everystep=false)
#
# plot(sol,vars=(1,2,3))
# plot(sol,vars=(0,2))
