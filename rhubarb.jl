using DifferentialEquations,LinearAlgebra,SparseArrays,BenchmarkTools
using Plots; plotly()

function l_coeffs(β,ν)

    ω = zeros(nx*(2*ny-1))
    # ω = [kx|ky ≠ 0 ? im*β*kx^2/(kx^2 + ky^2) : 0.0 for kx=spanx for ky=spany]

    v = zeros(nx*(2*ny-1))
    # v = [ComplexF64(ν*(kx^2 + ky^2)) for kx=spanx for ky=spany]

    return ω - v

end

function nl_coeffs(triads,pm)

    function pos(p)
        return p[1]*(2*ny-1) + (p[2] + ny)
    end

    Ks,Cs       = Int[],SparseMatrixCSC{ComplexF64}[]
    Ps,Qs,As    = Int[],Int[],ComplexF64[]

    for (i,triad) ∈ enumerate(triads)

        K,P,Q   = triad.k ,triad.p, triad.q

        kx,ky   = ComplexF64(K[1]),ComplexF64(K[2])
        px,py   = ComplexF64(P[1]),ComplexF64(P[2])
        qx,qy   = ComplexF64(Q[1]),ComplexF64(Q[2])
        Akpq    = pm*0.5*((px*qy - qx*py)*(qx^2 + qy^2 - px^2 - py^2)/(kx^2 + ky^2))

        push!(Ps,pos(P))
        push!(Qs,pos(Q))
        push!(As,Akpq)

        next = i < length(triads) ? triads[i + 1] : nothing
        if(next == nothing || next.k ≠ triad.k)

            # println("Building A for k-> ", pos(K))
            A = sparse(Ps,Qs,As,nx*(2*ny-1),nx*(2*ny-1))

            push!(Ks,pos(K))
            push!(Cs,A)

            Ps,Qs,As = Int[],Int[],ComplexF64[]

        end

    end

    return zip(Ks,Cs)

end

function nl_eqs!(du,u,p,t)

    function ind(p)
        return div(p-1,2*ny-1),mod(p-1,2*ny-1)-ny+1
    end

    nx, ny, Bωv, Cp, Cm  = p
    N = nx*(2*ny-1)

    Σp,Σm   = fill!(similar(u),0),fill!(similar(u),0)

    # modes: k = p + q
    for (k,c) ∈ Cp

        @show k,ind(k)
        temp = fill!(similar(u),0)

        # mul!(temp,c',u)
        # Σp[k] = dot(temp,u)
        # Σp[k] = dot(u,BLAS.gemv('T',c,u))

        mul!(temp,transpose(c),u)
        Σp[k] = BLAS.dotu(N,u,1,temp,1)

    end

    # modes: k = p - q
    for (k,c) ∈ Cm

        temp = fill!(similar(u),0)
        @show k,ind(k)

        mul!(temp,transpose(c),u)
        Σm[k] = BLAS.dotc(N,u,1,temp,1)

    end

    Σ       = BLAS.axpy!(1.0,Σp,Σm)
    temp    = BLAS.axpy!(1.0,Bωv.*u,Σ)

    # @show temp
    # du .= ωv.*u .+ Σ
    # du .= fill!(similar(u),0)
    @time du      .= temp

    spanx   = range(0,nx-1,step=1)
    spany   = range(-ny+1,ny-1,step=1)

    E = sum(ind(k)[1]|ind(k)[2] ≠ 0 ? abs(u[k])^2/(ind(k)[1]^2 + ind(k)[2]^2) : 0.0 for k ∈ 1:nx*(2*ny-1))
    Z = sum(abs(u[k])^2 for k ∈ 1:nx*(2*ny-1))

    @show t, E,Z

end

# resolution
nx,ny   = 2,2
spanx   = range(0,nx-1,step=1)
spany   = range(-ny+1,ny-1,step=1)

# linear coefficients
β,ν     = 1.0e-3,5e-3
Bωv     = l_coeffs(β,ν)

# non-linear coefficients
Δp      =  [(k=(kx,ky),p=(px,py),q=(qx,qy))
            for kx=spanx for ky=spany for px=spanx for py=spany for qx=spanx for qy=spany
            if (kx|ky ≠ 0 && px|py ≠ 0 && qx|qy ≠ 0)
            && (kx == px + qx && ky == py + qy)]

sort!(Δp,by=first)

Cp      = nl_coeffs(Δp,+1)

Δm      =  [(k = (kx,ky),p = (px,py), q = (qx,qy))
            for kx=spanx for ky=spany for px=spanx for py=spany for qx=spanx for qy=spany
            if (kx|ky ≠ 0 && px|py ≠ 0 && qx|qy ≠ 0)
            && (kx == px - qx && ky == py - qy)]

sort!(Δm,by=first)

@show Δp

@show Δm

Cm      = nl_coeffs(Δm,-1)

# setup and solve equations
u0      = randn(ComplexF64,nx*(2*ny-1))
tspan   = (0.0,20.0)
p       = [nx,ny,Bωv,Cp,Cm]

prob    = ODEProblem(nl_eqs!,u0,tspan,p)
sol     = @time solve(prob,Tsit5(),adaptive=true)

Plots.plot(sol,vars=(0,1),linewidth=4,label="(0,-1)",legend=true)
Plots.plot!(sol,vars=(0,2),linewidth=4,label="(0,0)",legend=true)
Plots.plot!(sol,vars=(0,3),linewidth=4,label="(0,1)",legend=true)
Plots.plot!(sol,vars=(0,4),linewidth=4,label="(1,-1)",legend=true)
Plots.plot!(sol,vars=(0,5),linewidth=4,label="(1,0)",legend=true)
Plots.plot!(sol,vars=(0,6),linewidth=4,label="(1,1)",legend=true)

# f(x,y)  = (x,abs(y)^2)
# Plots.plot(sol,vars=(f,1,2)linewidth=4,legend=true)
