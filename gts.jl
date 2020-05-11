using DifferentialEquations, LinearAlgebra
using StaticArrays, SparseArrays, BenchmarkTools
using Plots

function lorenz!(du,u,p,t)
    # Lorenz Equations
    x,y,z = u
    σ,ρ,β = p

    du[1] = dx = σ*(y-x)
    du[2] = dy = x*(ρ-z) - y
    du[3] = dz = x*y - β*z
end

function lv!(du,u,p,t)
    # Lotka-Volterra Equations
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α*x - β*x*y
    du[2] = dy = -δ*y + γ*x*y
end

function L(p)
    # calculate linear coefficient vector
    α, β, δ, γ = p

    coeffs = [α 0.0;
              0.0 -δ]

    return coeffs
end

function N(p)
    # calculate linear coefficient vector
    α, β, δ, γ = p

    coeffs = [0.0 -β;
              γ 0.0]

    return coeffs
end

function lv_generalised!(du,u,p,t)
    # Generalised Lotka-Volterra Equations
    x, y = u
    α, β, δ, γ = p

    # a = 0.0
    b = L(p)*u
    c = N(p) .* (u * u')

    du[1] = dx = b[1] + c[1,1] + c[1,2]
    du[2] = dy = b[2] + c[2,1] + c[2,2]
end

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

# LV
u0 = [1.0,1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(lv_generalised!,u0,tspan,p)
sol  = solve(prob,RK4(),adaptive=true)
plot(sol,vars=(1))
plot!(sol,vars=(2))
