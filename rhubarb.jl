using DifferentialEquations
using TimerOutputs
using Plots; plotly()

const Lx = 2.0*pi
const Ly = 2.0*pi

function nl_coeffs(X,Y,M,N)

    Δp = []
    Cp = Float64[]
    for m1 ∈ 0:1:M

        n1min = m1 == 0 ? 1 : -N
        for n1 ∈ n1min:1:N

            m2max = min(m1,M-m1)
            for m2 ∈ 0:1:m2max

                n2min = m2 == 0 ? 1 : -N
                for n2 ∈ n2min:1:N

                    m, n = m1 + m2, n1 + n2

                    if (m == 0 && n ∈ 1:1:N) || (m >= 1 && n ∈ -N:1:N)

                        push!(Δp,[m,n,m1,n1,m2,n2])

                        px,py   = (2.0*pi/X)*Float64(m1),(2.0*pi/Y)*Float64(n1)
                        qx,qy   = (2.0*pi/X)*Float64(m2),(2.0*pi/Y)*Float64(n2)

                        if m1 ≠ m2
                            c       = -(px*qy - qx*py)*(1.0/(px^2 + py^2) - 1.0/(qx^2 + qy^2))
                        else
                            c       = -(px*qy - qx*py)/(px^2 + py^2)
                        end
                        push!(Cp,c)

                        # println("[",m,",",n,"] = [",m1,",",n1,"] + [",m2,",",n2,"] -> ",c)

                    end

                end
            end
        end
    end

    Δm = []
    Cm = Float64[]
    for m1 ∈ 0:1:M

        n1min = m1 == 0 ? 1 : -N
        for n1 ∈ n1min:1:N

            for m2 ∈ 0:1:m1

                n2min = m2 == 0 ? 1 : -N
                for n2 ∈ n2min:1:N

                    m, n = m1 - m2, n1 - n2

                    if (m == 0 && n ∈ 1:1:N) || (m >= 1 && n ∈ -N:1:N)

                        push!(Δm,[m,n,m1,n1,m2,n2])

                        px,py   = (2.0*pi/X)*Float64(m1),(2.0*pi/Y)*Float64(n1)
                        qx,qy   = (2.0*pi/X)*Float64(m2),(2.0*pi/Y)*Float64(n2)

                        c       = (px*qy - qx*py)*(1.0/(px^2 + py^2) - 1.0/(qx^2 + qy^2))
                        push!(Cm,c)

                        # println("[",m,",",n,"] = [",m1,",",n1,"] + [",-m2,",",-n2,"] -> ",c)

                    end

                end
            end
        end
    end

    # @show Δp, Δm

    return zip(Δp,Cp),zip(Δm,Cm)

end

function nl_eqs!(du,u,p,t)

    nx::Int, ny::Int, Cp, Cm  = p

    dζ = fill!(similar(du),0)

    for (Δ,C) ∈ Cp

        m,n     = Δ[1] + 1, Δ[2] + ny
        m1,n1   = Δ[3] + 1, Δ[4] + ny
        m2,n2   = Δ[5] + 1, Δ[6] + ny

        dζ[m,n] += C*u[m1,n1]*u[m2,n2]

        # @show Δ, C
        # println("m = ", Δ[1],", n = ", Δ[2],", p = ", k, ", dζ += ", C*u[p]*u[q], "\n")

    end

    for (Δ,C) ∈ Cm

        m,n     = Δ[1] + 1, Δ[2] + ny
        m1,n1   = Δ[3] + 1, Δ[4] + ny
        m2,n2   = Δ[5] + 1, Δ[6] + ny

        dζ[m,n] += C*u[m1,n1]*conj(u[m2,n2])

        # @show Δ, C
        # println("m = ", Δ[1],", n = ", Δ[2],", p = ", k, ", dζ += ", C*u[p]*conj(u[q]), "\n")

    end

    du .= dζ

end

function opt_eqs()

    samples = 6
    timings = zeros(samples)
    for i in 1:samples

        nx = i + 1
        ny = i + 1

        println("Solving Nx2N system with N = ", nx)
        u0 = randn(ComplexF64,nx,(2*ny-1))
        tspan = (0.0,100.0)
        C1,C2 = nl_coeffs(Lx,Ly,nx-1,ny-1)
        p = [nx,ny,C1,C2]
        prob = ODEProblem(nl_eqs!,u0,tspan,p)
        timings[i] = @elapsed solve(prob,RK4(),adaptive=true,progress=true,save_start=false,save_everystep=false)

    end

    dims = [i + 1 for i in 1:samples]
    Plots.plot(dims,timings,scale=:log,xaxis="N",yaxis="T",markershape = :square,legend=false)

end

function exec()

    lx::Float64 = 2.0*Float64(pi)
    ly::Float64 = 2.0*Float64(pi)
    nx::Int = 4
    ny::Int = 4

    u0 = randn(ComplexF64,nx,(2*ny-1))
    tspan = (0.0,100.0)
    C1,C2 = nl_coeffs(lx,ly,nx-1,ny-1)
    p = [nx,ny,C1,C2]
    prob = ODEProblem(nl_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=100,save_start=true,saveat=20,save_everystep=false)
    # integrator = init(prob,RK4())
    # step!(integrator)

    return sol
end

opt_eqs()

sol = exec()

du = similar(u0)
@time nl_eqs!(du,u0,p,tspan)
@code_warntype nl_eqs!(du,u0,p,tspan)

Plots.plot(sol,vars=(0,1),linewidth=2,label="(0,-1)",legend=true)
Plots.plot!(sol,vars=(0,2),linewidth=2,label="(0,0)")
Plots.plot!(sol,vars=(0,3),linewidth=2,label="(0,1)")
Plots.plot!(sol,vars=(0,4),linewidth=2,label="(1,-1)")
Plots.plot!(sol,vars=(0,5),linewidth=2,label="(1,0)")
Plots.plot!(sol,vars=(0,6),linewidth=2,label="(1,1)")

E = zeros(Float64,length(sol.u))
Z = zeros(Float64,length(sol.u))

for (i,u) ∈ enumerate(sol.u)

    for j ∈ 0:1:nx-1
        kmin = j == 0 ? 1 : -ny + 1
        for k ∈ kmin:1:ny-1

            m,n   = j + 1, k + ny

            cx,cy = (2.0*pi/Lx)*j,(2.0*pi/Ly)*k

            E[i] += abs(u[m,n])^2/(cx^2 + cy^2)
            Z[i] += abs(u[m,n])^2

        end
    end
end

Plots.plot(sol.t,E,linewidth=2,legend=true,xaxis="t",label="E (Energy)")
Plots.plot!(sol.t,Z,linewidth=2,legend=true,xaxis="t",label="Z (Enstrophy)",yaxis="E, Z")

@show Z
