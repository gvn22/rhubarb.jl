using DifferentialEquations,LinearAlgebra,SparseArrays,BenchmarkTools
using Plots; plotly()

function l_coeffs(β,ν)

    ω = zeros(nx*(2*ny-1))
    # ω = [kx|ky ≠ 0 ? im*β*kx^2/(kx^2 + ky^2) : 0.0 for kx=spanx for ky=spany]

    v = zeros(nx*(2*ny-1))
    # v = [ComplexF64(ν*(kx^2 + ky^2)) for kx=spanx for ky=spany]

    return ω - v

end

function nl_coeffs(X,Y,M,N)

    function pos(p)
        return p[1]*(2*N+1) + (p[2] + N + 1)
    end

    cx,cy = 2.0*pi/X,2.0*pi/Y

    Δppq = [(k=(kx,ky),p=(px,py),q=(qx,qy))
            for kx=0:M for ky=-N:N
            for px=0:M for py=-N:N
            for qx=0:M for qy=-N:N
            if (kx|ky ≠ 0 && px|py ≠ 0 && qx|qy ≠ 0)
            && (kx == px + qx && ky == py + qy)]

    println("Cppq...")
    Cppq = Float64[]
    for Δ ∈ Δppq

        k,p,q   = pos(Δ.k),pos(Δ.p),pos(Δ.q)
        kx,ky   = cx*Float64(Δ.k[1]),cy*Float64(Δ.k[2])
        px,py   = cx*Float64(Δ.p[1]),cy*Float64(Δ.p[2])
        qx,qy   = cx*Float64(Δ.q[1]),cy*Float64(Δ.q[2])

        # c = 0.5*(px*qy - qx*py)*((qx^2 + qy^2) - (px^2 + py^2))/(kx^2 + ky^2)
        c = (px*qy - qx*py)/(px^2 + py^2)

        push!(Cppq,c)
        @show Δ.k,Δ.p,Δ.q,c

    end

    Δpmq = [(k=(kx,ky),p=(px,py),q=(qx,qy))
            for kx=0:M for ky=-N:N
            for px=1:M for py=-N:N
            for qx=1:M for qy=-N:N
            if (kx|ky ≠ 0 && px|py ≠ 0 && qx|qy ≠ 0)
            && (kx == px - qx && ky == py - qy)]

    println("Cpmq...")
    Cpmq = Float64[]
    for Δ ∈ Δpmq

        k,p,q = pos(Δ.k),pos(Δ.p),pos(Δ.q)
        kx,ky   = cx*Float64(Δ.k[1]),cy*Float64(Δ.k[2])
        px,py   = cx*Float64(Δ.p[1]),cy*Float64(Δ.p[2])
        qx,qy   = cx*Float64(Δ.q[1]),cy*Float64(Δ.q[2])

        # c = - 0.5*(px*qy - qx*py)*((qx^2 + qy^2) - (px^2 + py^2))/(kx^2 + ky^2)
        c = - (px*qy - qx*py)/(px^2 + py^2)

        push!(Cpmq,c)
        @show Δ.k,Δ.p,Δ.q,c

    end

    Δmpq = [(k=(kx,ky),p=(px,py),q=(qx,qy))
            for kx=0:M for ky=-N:N
            for px=1:M for py=-N:N
            for qx=1:M for qy=-N:N
            if (kx|ky ≠ 0 && px|py ≠ 0 && qx|qy ≠ 0)
            && (kx == - px + qx && ky == - py + qy)]

    println("Cmpq...")
    Cmpq = Float64[]
    for Δ ∈ Δmpq

        k,p,q   = pos(Δ.k),pos(Δ.p),pos(Δ.q)
        kx,ky   = cx*Float64(Δ.k[1]),cy*Float64(Δ.k[2])
        px,py   = cx*Float64(Δ.p[1]),cy*Float64(Δ.p[2])
        qx,qy   = cx*Float64(Δ.q[1]),cy*Float64(Δ.q[2])

        # c = - 0.5*(px*qy - qx*py)*((qx^2 + qy^2) - (px^2 + py^2))/(kx^2 + ky^2)
        c = - (px*qy - qx*py)/(px^2 + py^2)

        push!(Cmpq,c)
        @show  Δ.k,Δ.p,Δ.q,c

    end

    # @show Δ1
    # @show Δ2
    # @show Δ3
    return zip(Δppq,Cppq),zip(Δpmq,Cpmq),zip(Δmpq,Cmpq)

end

function nl_eqs!(du,u,p,t)

    X, Y, nx, ny, C1, C2, C3  = p

    function ind(x)
        return div(x-1,2*ny-1),mod(x-1,2*ny-1)-ny+1
    end

    dkx,dky = 2.0*pi/X,2.0*pi/Y

    # E = sum(abs(u[k])^2*((ind(k)[1]*dkx)^2 + (ind(k)[2]*dky)^2) for k ∈ 1:nx*(2*ny-1))
    # Z = sum(abs(u[k])^2 for k ∈ 1:nx*(2*ny-1))
    #
    # @show t, E,Z

    N = ny - 1
    function pos(p)
        return p[1]*(2*N+1) + (p[2] + N + 1)
    end

    dψ = fill!(similar(u),0)
    # @show du
    for (Δ,C) ∈ C1

        k,p,q   = pos(Δ.k),pos(Δ.p),pos(Δ.q)
        dψ[k]   += C*u[p]*u[q]

    end

    for (Δ,C) ∈ C2

        k,p,q   = pos(Δ.k),pos(Δ.p),pos(Δ.q)
        dψ[k]   += C*u[p]*conj(u[q])

    end

    for (Δ,C) ∈ C3

        k,p,q   = pos(Δ.k),pos(Δ.p),pos(Δ.q)
        dψ[k]   += C*conj(u[p])*u[q]

    end

    # @show t,dψ
    du .= dψ

end

Lx,Ly   = 2.0*pi,2.0*pi
nx,ny   = 2,2

# ωv      = lin_coeffs()
C1,C2,C3= nl_coeffs(Lx,Ly,nx-1,ny-1)

u0      = rand(ComplexF64,nx*(2*ny-1))
tspan   = (0.0,100.0)
p       = [Lx,Ly,nx,ny,C1,C2,C3]

prob    = ODEProblem(nl_eqs!,u0,tspan,p)
sol     = solve(prob,RK4(),adaptive=true,progress=true)
# integrator = init(prob,RK4())
# step!(integrator)

Plots.plot(sol,vars=(0,1),linewidth=2,label="(0,-1)",legend=true)
Plots.plot!(sol,vars=(0,2),linewidth=2,label="(0,0)")
Plots.plot!(sol,vars=(0,3),linewidth=2,label="(0,1)")
Plots.plot!(sol,vars=(0,4),linewidth=2,label="(1,-1)")
Plots.plot!(sol,vars=(0,5),linewidth=2,label="(1,0)")
Plots.plot!(sol,vars=(0,6),linewidth=2,label="(1,1)")

# f(x,y)  = (x,abs(y)^2)
# Plots.plot(sol,vars=(f,0,1),linewidth=2,legend=true)
# Plots.plot!(sol,vars=(f,0,2),linewidth=2,legend=true)
# Plots.plot!(sol,vars=(f,0,3),linewidth=2,legend=true)
# Plots.plot!(sol,vars=(f,0,4),linewidth=2,legend=true)
# Plots.plot!(sol,vars=(f,0,5),linewidth=2,legend=true)
# Plots.plot!(sol,vars=(f,0,6),linewidth=2,legend=true)
#
#
#
cx,cy = 2.0*pi/Lx,2.0*pi/Ly
function ind(x)
    return div(x-1,2*ny-1),mod(x-1,2*ny-1)-ny+1
end

E = [sum(abs(u[k])^2/((ind(k)[1]*cx)^2 + (ind(k)[2]*cy)^2) for k=1:1:(nx-1)*(2*ny-1) if ind(k)[1]|ind(k)[2] ≠ 0) for u in sol.u]
Plots.plot(E,linewidth=2,legend=true)

Z = [sum(abs(u[k])^2 for k=1:1:(nx-1)*(2*ny-1)) for u in sol.u]
Plots.plot(Z,linewidth=2,legend=true)
