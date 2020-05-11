using DifferentialEquations, LinearAlgebra
using StaticArrays, SparseArrays, BenchmarkTools
using Plots

function lorenz!(du,u,p,t)
    x,y,z = u
    σ,ρ,β = p

    du[1] = dx = σ*(y-x)
    du[2] = dy = x*(ρ-z) - y
    du[3] = dz = x*y - β*z
end

u0 = [1.0;0.0;0.0]
tspan = (0.0,100.0)
p = [10.0,28.0,8/3]

prob = ODEProblem(lorenz!,u0,tspan,p)
# sol = solve (prob)
@benchmark sol = solve(prob,Tsit5(),save_everystep=false)

plot(sol,vars=(1,2,3))
plot(sol,vars=(0,2))
