using DifferentialEquations, LinearAlgebra
using StaticArrays, SparseArrays
using BenchmarkTools, Test
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
    return @SMatrix [α 0.0;
                    0.0 -δ]

end

function N(p)

    β, γ = p
    return @SMatrix [0.0 -β;
             γ 0.0]

end

function F(p)

    τ = p
    return @SMatrix [0.0 0.0;
                       0.0 0.0]

end

#! solve the equation dq = L(q) + N(q,q) + F(q)
function glv!(du,u,p,t)

    # Generalise LV towards GQL
    α, β, δ, γ, τ = p

    # a = 0.0
    # b = copy(u)
    b = L([α,δ]) * u

    # c = copy()
    c = N([β,γ]) .* (u*u')
    # d = F([τ]) * u

    for i ∈ [1,2]
        du[i] = b[i] + sum(c[i,:])
    end

end

function nl!(du,u,p,t)

    β, ν = p

    # Rossby
    b = [β*k^2/(k^2 + l^2) for k in 1:3, l in 1:3]
    du .= b.*u

end

u0 = rand(3,3)
p = [1.0,1.0]

tspan   = (0.0,1.0)
prob = ODEProblem(nl!,u0,tspan,p)
sol  = solve(prob,RK4(),adaptive=true)

plot(sol)
