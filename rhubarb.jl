using DifferentialEquations
using Plots
using Einsum

function nl_eqs!(du,u,p,t)

    # parameters
    nx, ny, β, ν  = p
    nx,ny = Int64(nx), Int64(ny)

    spanx = 0:1:nx-1
    spany = -ny+1:1:ny-1

    # linear coefficients
    ω = [kx|ky ≠ 0 ? im*β*kx^2/(kx^2 + ky^2) : 0.0 + im*0.0 for kx=spanx, ky=spany]
    v = [ComplexF64(ν*(kx^2 + ky^2)) for kx=spanx, ky=spany]

    # non-linear coefficients
    A = zeros(ComplexF64,nx,2*ny-1,nx,2*ny-1,nx,2*ny-1)

    triads = [(kx,ky,px,py,qx,qy) for kx=spanx, ky=spany, px=spanx, py=spany, qx=spanx, qy=spany
                if (kx|ky ≠ 0 && px|py ≠ 0 && qx|qy ≠ 0)
                && ((kx == px + qx && ky == py + qy) || (kx == px - qx && ky == py - qy))]

    for triad ∈ triads

        kx,ky,px,py,qx,qy = [Float64(triad[i]) for i=1:6]
        kpq = CartesianIndex(triad) + CartesianIndex(1,ny,1,ny,1,ny)
        A[kpq] = -1.0/(4.0*pi)*(px*qy - py*qx)*(px^2 + py^2 - qx^2 - qy^2)/(kx^2 + ky^2)

    end

    # tendency equation (optimise later using sparsearrays)
    @einsum du[i,j] = - ω[i,j]*u[i,j] - v[i,j]*u[i,j] + A[i,j,k,l,m,n]*u[k,l]*u[m,n]

end

u0 = rand(ComplexF64,nx,2*ny-1)
p = [2,2,1.0e-4,1.5e-5]

tspan = (0.0,1000.0)
prob = ODEProblem(nl_eqs!,u0,tspan,p)
sol  = solve(prob,Tsit5(),adaptive=true,stiff=true)

plot(sol,linewidth=1,legend=false)
