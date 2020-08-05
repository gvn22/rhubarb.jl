using DifferentialEquations,LinearAlgebra
using Plots; plotly()

function coeffs(X,Y,M,N)

    cx,cy = 2.0*pi/X,2.0*pi/Y

    println("Cp...")

    Δp = []
    Cp = Float64[]
    for m1 ∈ 0:1:M

        n1min = m1 == 0 ? 1 : -N
        for n1 ∈ n1min:1:N

            for m2 ∈ 0:1:M-m1

                n2min = m2 == 0 ? 1 : -N
                for n2 ∈ n2min:1:N

                    if !(m1|n1 == 0) && !(m2|n2 == 0)

                        m = m1 + m2
                        n = n1 + n2

                        if (m >= 0 && m <= M) && (n >= -N && n <= N) && !(m|n == 0)

                            push!(Δp,[m,n,m1,n1,m2,n2])

                            c = - (cx*m1*cy*n2 - cx*m2*cy*n1)/(cx^2*m1^2 + cy^2*n1^2)
                            push!(Cp,c)

                            println("[",m,",",n,"] = [",m1,",",n1,"] + [",m2,",",n2,"] -> ",c)

                        end

                    end

                end
            end
        end
    end

    @show Δp

    println("Cm...")

    Δm = []
    Cm = Float64[]
    for m1 ∈ 0:1:M

        n1min = m1 == 0 ? 1 : -N
        for n1 ∈ n1min:1:N

            for m2 ∈ 0:1:m1

                n2min = m2 == 0 ? 1 : -N
                for n2 ∈ n2min:1:N

                    if !(m1|n1 == 0) && !(m2|n2 == 0)

                        m = m1 - m2
                        n = n1 - n2

                        if (m >= 0 && m <= M) && (n >= -N && n <= N) && !(m == 0 && n <= 0)

                            push!(Δm,[m,n,m1,n1,m2,n2])

                            c = (cx*m1*cy*n2 - cx*m2*cy*n1)/(cx^2*m1^2 + cy^2*n1^2)
                            d = (cx*m2*cy*n1 - cx*m1*cy*n2)/(cx^2*m2^2 + cy^2*n2^2)

                            println("[",m,",",n,"] = [",m1,",",n1,"] + [",-m2,",",-n2,"] -> ",c," ",d)
                            push!(Cm,c+d)

                        end
                    end

                end
            end
        end
    end

    @show Δm

    return zip(Δp,Cp),zip(Δm,Cm)

end

function nl_eqs!(du,u,p,t)

    X, Y, nx, ny, Cp, Cm  = p

    function pos(a,b)
        return a*(2*ny-1) + (b + ny)
    end

    dζ = fill!(similar(u),0)

    println("Cp...")

    for (Δ,C) ∈ Cp

        @show Δ, C
        k,p,q     = pos(Δ[1],Δ[2]),pos(Δ[3],Δ[4]),pos(Δ[5],Δ[6])

        dζ[k]   = dζ[k] + C*u[p]*u[q]
        println("m = ", Δ[1],", n = ", Δ[2],", p = ", k, ", dζ += ", C*u[p]*u[q], "\n")

    end

    println("Cm...")

    for (Δ,C) ∈ Cm

        @show Δ, C

        k,p,q   = pos(Δ[1],Δ[2]),pos(Δ[3],Δ[4]),pos(Δ[5],Δ[6])

        dζ[k]   = dζ[k] + C*u[p]*conj(u[q])
        println("m = ", Δ[1],", n = ", Δ[2],", p = ", k, ", dζ += ", C*u[p]*conj(u[q]), "\n")

        # @show dζ[k]

    end

    @show dζ
    du .= dζ
    du[1] = du[3]

end

Lx,Ly   = 2.0*pi,2.0*pi
nx,ny   = 2,2

C1,C2   = coeffs(Lx,Ly,nx-1,ny-1)

# u0      = randn(ComplexF64,nx*(2*ny-1))

u0      = [1.0,0.0,1.0,2.0,3.0,4.0]
tspan   = (0.0,100.0)
p       = [Lx,Ly,nx,ny,C1,C2]

prob    = ODEProblem(nl_eqs!,u0,tspan,p)
sol     = solve(prob,RK4(),adaptive=true,progress=true,reltol=1e-6,abstol=1e-6)
# integrator = init(prob,RK4())
# step!(integrator)

Plots.plot(sol,vars=(0,1),linewidth=2,label="(0,-1)",legend=true)
Plots.plot!(sol,vars=(0,2),linewidth=2,label="(0,0)")
Plots.plot!(sol,vars=(0,3),linewidth=2,label="(0,1)")
Plots.plot!(sol,vars=(0,4),linewidth=2,label="(1,-1)")
Plots.plot!(sol,vars=(0,5),linewidth=2,label="(1,0)")
Plots.plot!(sol,vars=(0,6),linewidth=2,label="(1,1)")

cx,cy = 2.0*pi/Lx,2.0*pi/Ly

function ind(x)
    return div(x-1,2*ny-1),mod(x-1,2*ny-1)-ny+1
end

E = zeros(Float64,length(sol.u))
Z = zeros(Float64,length(sol.u))

for (i,u) ∈ enumerate(sol.u)

    for k ∈ ny+1:1:nx*(2*ny-1)

        m,n   = ind(k)
        if !(m|n == 0)

            E[i] += abs(u[k])^2/(m^2 + n^2)
            Z[i] += abs(u[k])^2
        end

    end
end

Plots.plot(sol.t,E,linewidth=2,legend=true,xaxis="t",label="E (Energy)")
Plots.plot!(sol.t,Z,linewidth=2,legend=true,xaxis="t",label="Z (Enstrophy)",yaxis="E, Z")

@show sol.u
