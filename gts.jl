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
    # calculate linear coefficient vector
    α, β, δ, γ, τ = p

    coeffs = [α 1.0;
              0.0 -δ]

    return coeffs
end

function N(p)
    # calculate linear coefficient vector
    α, β, δ, γ, τ = p

    coeffs = [0.0 -β;
              γ 0.0]

    return coeffs
end

function F(p)
    # calculate a linear forcing term
    α, β, δ, γ, τ = p

    # start with 0 forcing
    coeffs = (0.0/τ)*[1.0 0.0;
                0.0 1.0]

    return coeffs
end

function glv!(du,u,p,t)
    # Generalisation of Lotka-Volterra -> towards GQL
    # x, y = u
    α, β, δ, γ, τ = p

    # a = 0.0
    b = L(p) * u
    c = N(p) .* (u * u')
    d = F(p) * u

    for i ∈ [1,2]
        du[i] = b[i] + sum(c[i,:])
        # du[2] = dy = b[2] + c[2,1] + c[2,2]
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
