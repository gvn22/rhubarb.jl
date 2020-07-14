using DifferentialEquations
using Plots, PyPlot
using Einsum, LinearAlgebra

function l_coeffs(β,ν,spanx,spany)

    # linear coefficients
    # ω = fill!(similar(u),0)
    ω = [kx|ky ≠ 0 ? im*β*kx^2/(kx^2 + ky^2) : 0.0 for ky=spany for kx=spanx]

    # v = fill!(similar(u),0)
    # v = [ComplexF64(ν*(kx^2 + ky^2)) for ky=spany for kx=spanx]

    return ω

end

function nl_coeffs(triads)

    # println("Computing coefficients...")
    function pos(p)
        return p[1]*(2*ny-1) + (p[2] + ny)
    end

    Ks  = Int[]
    Cs  = SparseMatrixCSC{ComplexF64}[]

    Ps  = Int[]
    Qs  = Int[]
    As  = ComplexF64[]
    for (i,triad) ∈ enumerate(triads)

        kx,ky   = ComplexF64(triad.k[1]),ComplexF64(triad.k[2])
        px,py   = ComplexF64(triad.p[1]),ComplexF64(triad.p[2])
        qx,qy   = ComplexF64(triad.q[1]),ComplexF64(triad.q[2])
        # akpq    = -0.5*((kx*py - ky*px)/(px^2 + py^2) + (kx*qy - ky*qx)/(qx^2 + qy^2))
        akpq    = 0.5*((px*qy - qx*py)*(qx^2 + qy^2 - px^2 - py^2)/(kx^2 + ky^2))

        P,Q = triad.p, triad.q

        push!(Ps,pos(P))
        push!(Qs,pos(Q))
        push!(As, akpq)

        next = i < length(triads) ? triads[i + 1] : nothing
        if(next == nothing || next.k ≠ triad.k)

            A = sparse(Ps,Qs,As,nx*(2*ny-1),nx*(2*ny-1))

            Ps = Int[]
            Qs = Int[]
            As = ComplexF64[]

            K = triad.k
            push!(Ks,pos(K))
            push!(Cs,A)

        end

    end

    return Ks,Cs

end

function nl_eqs!(du,u,p,t)

    nx, ny, β, ν  = p
    nx,ny = Int64(nx), Int64(ny)
    spanx   = range(0,nx-1,step=1)
    spany   = range(-ny+1,ny-1,step=1)

    E = sum(kx|ky ≠ 0 ? u[kx*(2*ny - 1) + ky+ny]^2/(kx^2 + ky^2) : 0.0 for ky=spany for kx=spanx)
    Z = sum(kx|ky ≠ 0 ? u[kx*(2*ny - 1) + ky+ny]^2 : 0.0 for ky=spany for kx=spanx)

    @show E,Z

    # modes: k = p + q
    Bp      = ComplexF64[]
    temp    = similar(u)

    for (mode,matrix) in zip(Kp,Cp)

        mul!(temp,matrix,u)
        push!(Bp,dot(temp,u))

    end

    Σp = sparsevec(Kp,Bp,(nx)*(2*ny-1))
    # @show Kp,Σp

    # modes: k = p - q
    Bm      = ComplexF64[]
    temp    = similar(u)

    for (mode,matrix) in zip(Km,Cm)

        mul!(temp,matrix,conj!(u))
        push!(Bm,dot(temp,u))

    end

    Σm = sparsevec(Km,Bm,(nx)*(2*ny-1))
    # @show Km,Σm

    du .= Σp + Σm

    # q = u + β
    # @show Γ2

end

nx,ny   = 3,3
β,ν     = 1.0e-3,5e-3
p       = [nx,ny,β,ν]

X,Y     = nx,2*ny - 1
u0      = randn(ComplexF64,X*Y)

tspan   = (0.0,10.0)

spanx   = range(0,nx-1,step=1)
spany   = range(-ny+1,ny-1,step=1)

Δp      =  [(k=(kx,ky),p=(px,py),q=(qx,qy))
            for qx=spanx, qy=spany, px=spanx, py=spany, kx=spanx, ky=spany
            if (kx|ky ≠ 0 && px|py ≠ 0 && qx|qy ≠ 0)
            && (kx == px + qx && ky == py + qy)]

Kp,Cp   = nl_coeffs(Δp)
# @show Δp,Kp

Δm      =  [(k = (kx,ky),p = (px,py), q = (qx,qy))
            for qx=spanx, qy=spany, px=spanx, py=spany, kx=spanx, ky=spany
            if (kx|ky ≠ 0 && px|py ≠ 0 && qx|qy ≠ 0)
            && (kx == px - qx && ky == py + qy)]

Km,Cm   = nl_coeffs(Δm)
# @show Δm,Km

allΔs   = vcat(Δp,Δm)
# @show unique(x->x.k,tri_adds)

@show Δp

@show Δm
prob    = ODEProblem(nl_eqs!,u0,tspan,p)
sol     = solve(prob,Tsit5(),adaptive=true)

# pyplot()
# plot(sol,linewidth=1,legend=false)
