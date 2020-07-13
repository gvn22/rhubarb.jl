using DifferentialEquations
using Plots, PyPlot
using Einsum, LinearAlgebra

function nl_eqs!(du,u,p,t)

    # parameters
    nx, ny, β, ν  = p
    nx,ny = Int64(nx), Int64(ny)

    spanx = range(0,nx-1,step=1)
    spany = range(-ny+1,ny-1,step=1)

    # forcing
    # γ = []

    # linear coefficients
    # ω = zeros(ComplexF64,nx,2*ny-1)
    # ω = [kx|ky ≠ 0 ? im*β*kx^2/(kx^2 + ky^2) : 0.0 + im*0.0 for kx=spanx, ky=spany]

    # v = zeros(ComplexF64,nx,2*ny-1)
    # v = [ComplexF64(ν*(kx^2 + ky^2)) for kx=spanx, ky=spany]

    # non-linear coefficients
    # A = zeros(Float64,nx,2*ny-1,nx,2*ny-1,nx,2*ny-1)

    # triads obeying selection rules
    # triads = [(kx,ky,px,py,qx,qy) for kx=spanx, ky=spany, px=spanx, py=spany, qx=spanx, qy=spany
    #             if (kx|ky ≠ 0 && px|py ≠ 0 && qx|qy ≠ 0)
    #             && ((kx == px + qx && ky == py + qy) || (kx == px - qx && ky == py - qy))]
    println("Constructing set of triads...")

    tri_adds =  [(k = (kx,ky),p = (px,py), q = (qx,qy))
                for qx=spanx, qy=spany, px=spanx, py=spany, kx=spanx, ky=spany
                if (kx|ky ≠ 0 && px|py ≠ 0 && qx|qy ≠ 0)
                && (kx == px + qx && ky == py + qy)]

    tri_difs =  [(k = (kx,ky),p = (px,py), q = (qx,qy))
                for qx=spanx, qy=spany, px=spanx, py=spany, kx=spanx, ky=spany
                if (kx|ky ≠ 0 && px|py ≠ 0 && qx|qy ≠ 0)
                && (kx == px - qx && ky == py - qy)]

    triads = vcat(tri_adds,tri_difs)
    # @show length(triads),3*5*3*5*3*5*2

    # for triad ∈ triads
    #
    #     kx,ky,px,py,qx,qy = [Float64(mode) for mode ∈ triad]
    #     kpq = CartesianIndex(triad) + CartesianIndex(1,ny,1,ny,1,ny)
    #
    #     # kx,ky = assign here rather than use indices
    #     A[kpq] = -1.0/(100.0)*(px*qy - py*qx)*(px^2 + py^2 - qx^2 - qy^2)/(kx^2 + ky^2)
    #
    # end
    #
    # c = zeros(ComplexF64,nx,2*ny-1)
    # @einsum c[i,j] := A[i,j,k,l,m,n]*u[k,l]*u[m,n]
    #
    # @show t
    # # @show size(du), size(ω), size(v), size(A)
    #
    # # tendency equation (to be optimised using sparsearrays+BLAS)
    # du .= - ω .* u - v .* u + c

end

# compute absolute vorticity
# function q(ψ):
    # β
# end

# initialise problem
nx,ny = 3,3

u0      = randn(ComplexF64,nx,2*ny-1)
p       = [nx,ny,1.0e-2,5e-3]
tspan   = (0.0,1.0)

# choose equations and solve
prob = ODEProblem(nl_eqs!,u0,tspan,p)
sol  = solve(prob,RK4(),adaptive=false,dt=0.01,maxiters=10)

# concatenate conjugate matrix
# transforum to spatial coordinates

# plot
# pyplot()
# plot(sol,linewidth=1,legend=false)
