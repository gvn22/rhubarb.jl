using DifferentialEquations, LinearAlgebra
using StaticArrays, SparseArrays
using BenchmarkTools, Test
using Plots
using Einsum

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

    nx = 3
    ny = 3

    # Rossby + diffusion
    b = [im*β*k^2/(k^2 + l^2) - ν*(k^2 + l^2) for k in 1:nx, l in 1:ny]
    b[1,1] = 0.0

    c = zeros(ComplexF64,(nx,ny,nx,ny,nx,ny))
    for k in 1:nx
        for l in 1:ny
            for m in 1:nx
                for n in 1:ny

                    i = k + m
                    j = l + n

                    if i <= nx && j <= ny
                        c[i,j,k,l,m,n] = - Float64(((k^2 + l^2)^2 - (m^2 + n^2)^2)/(i^2 + j^2))
                        # @show i,j,k,l,m,n,c[i,j,k,l,m,n]
                    end

                    i = k - m
                    j = l - n

                    if i > 1 && j > 1
                        c[i,j,k,l,m,n] = - Float64(((k^2 + l^2)^2 - (m^2 + n^2)^2)/(i^2 + j^2))
                        # @show i,j,k,l,m,n
                    end

                end
            end
        end
    end

    # @show c
    du .= b .* u
    # @einsum du[i,j] = b[i,j]*u[i,j] + c[i,j,k,l,m,n]*u[k,l]*u[m,n]
    @einsum du[i,j] += c[i,j,k,l,m,n]*u[k,l]*u[m,n]

end

nx = 3
ny = 3

u0 = rand(ComplexF64,(nx,ny))
p = [1.0,0.0001]

tspan   = (0.0,1000.0)
prob = ODEProblem(nl!,u0,tspan,p)
sol  = solve(prob,RK4(),adaptive=true)

@show size(sol)[3]
plot(sol)
r = [(sol[2,1,i]) for i in 1:size(sol)[3]]
plot(r)
