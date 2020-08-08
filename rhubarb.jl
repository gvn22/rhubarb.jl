using DifferentialEquations
using TimerOutputs
using Plots; plotly()

function nl_coeffs(lx::Float64,ly::Float64,nx::Int,ny::Int)

    X = lx
    Y = ly
    M = nx - 1
    N = ny - 1

    Cp = zeros(Float64,nx,2*ny-1,nx,2*ny-1)
    Cm = zeros(Float64,nx,2*ny-1,nx,2*ny-1)

    for m1 ∈ 0:1:M

        n1min = m1 == 0 ? 1 : -N
        for n1 ∈ n1min:1:N

            m2max = min(m1,M-m1)
            for m2 ∈ 0:1:m2max

                n2min = m2 == 0 ? 1 : -N
                for n2 ∈ n2min:1:N

                    m, n = m1 + m2, n1 + n2

                    if (m == 0 && n ∈ 1:1:N) || (m >= 1 && n ∈ -N:1:N)

                        px,py   = (2.0*pi/X)*Float64(m1),(2.0*pi/Y)*Float64(n1)
                        qx,qy   = (2.0*pi/X)*Float64(m2),(2.0*pi/Y)*Float64(n2)

                        if m1 ≠ m2
                            c       = -(px*qy - qx*py)*(1.0/(px^2 + py^2) - 1.0/(qx^2 + qy^2))
                        else
                            c       = -(px*qy - qx*py)/(px^2 + py^2)
                        end

                        Cp[m1+1,n1+ny,m2+1,n2+ny] = c

                    end

                end
            end
        end
    end

    for m1 ∈ 0:1:M

        n1min = m1 == 0 ? 1 : -N
        for n1 ∈ n1min:1:N

            for m2 ∈ 0:1:m1

                n2min = m2 == 0 ? 1 : -N
                for n2 ∈ n2min:1:N

                    m, n = m1 - m2, n1 - n2

                    if (m == 0 && n ∈ 1:1:N) || (m >= 1 && n ∈ -N:1:N)

                        px,py   = (2.0*pi/X)*Float64(m1),(2.0*pi/Y)*Float64(n1)
                        qx,qy   = (2.0*pi/X)*Float64(m2),(2.0*pi/Y)*Float64(n2)

                        c       = (px*qy - qx*py)*(1.0/(px^2 + py^2) - 1.0/(qx^2 + qy^2))

                        Cm[m1+1,n1+ny,m2+1,n2+ny] = c

                    end

                end
            end
        end
    end

    return Cp,Cm
end

function nl_eqs!(du,u,p,t)

    nx::Int,ny::Int,Cp::Array{Float64,4},Cm::Array{Float64,4} = p
    M = nx - 1
    N = ny - 1

    dζ = fill!(similar(du),0)

    for m1 ∈ 0:1:M

        n1min = m1 == 0 ? 1 : -N
        for n1 ∈ n1min:1:N

            m2max = min(m1,M-m1)
            for m2 ∈ 0:1:m2max

                n2min = m2 == 0 ? 1 : -N
                for n2 ∈ n2min:1:N

                    m, n = m1 + m2, n1 + n2

                    if (m == 0 && n ∈ 1:1:N) || (m >= 1 && n ∈ -N:1:N)

                        dζ[m+1,n+ny] += Cp[m1+1,n1+ny,m2+1,n2+ny]*u[m1+1,n1+ny]*u[m2+1,n2+ny]

                    end

                end
            end
        end
    end

    for m1 ∈ 0:1:M

        n1min = m1 == 0 ? 1 : -N
        for n1 ∈ n1min:1:N

            for m2 ∈ 0:1:m1

                n2min = m2 == 0 ? 1 : -N
                for n2 ∈ n2min:1:N

                    m, n = m1 - m2, n1 - n2

                    if (m == 0 && n ∈ 1:1:N) || (m >= 1 && n ∈ -N:1:N)

                        dζ[m+1,n+ny] += Cm[m1+1,n1+ny,m2+1,n2+ny]*u[m1+1,n1+ny]*conj(u[m2+1,n2+ny])

                    end

                end
            end
        end
    end

    du .= dζ
    dζ = nothing
end

function opt_eqs()

    samples = 20
    timings = zeros(samples)
    for i in 1:2:samples

        nx = i + 1
        ny = i + 1

        println("Solving Nx2N system with N = ", nx)
        u0 = randn(ComplexF64,nx,2*ny-1)
        tspan = (0.0,100.0)
        Cp,Cm = nl_coeffs(lx,ly,nx,ny)
        p = [nx,ny,Cp,Cm]
        prob = ODEProblem(nl_eqs!,u0,tspan,p)
        timings[i] = @elapsed solve(prob,RK4(),adaptive=true,progress=true,save_start=false,save_everystep=false)

    end

    dims = [i + 1 for i in 1:samples]
    Plots.plot(dims,timings,scale=:log,xaxis="N",yaxis="T",markershape = :square,legend=false)

end

function exec(lx::Float64,ly::Float64,nx::Int,ny::Int)

    u0 = randn(ComplexF64,nx,(2*ny-1))
    tspan = (0.0,100.0)
    Cp,Cm = nl_coeffs(lx,ly,nx,ny)
    p = [nx,ny,Cp,Cm]
    prob = ODEProblem(nl_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=100,save_start=true,save_everystep=false)
    # integrator = init(prob,RK4())
    # step!(integrator)

    return sol
end

function energy(lx,ly,nx,ny,sol)

    E = zeros(Float64,length(sol.u))
    Z = fill!(similar(E),0)

    for i in eachindex(sol.u)

        for j ∈ 0:1:nx-1
            kmin = j == 0 ? 1 : -ny + 1
            for k ∈ kmin:1:ny-1

                m,n   = j + 1, k + ny

                cx,cy = (2.0*pi/Lx)*j,(2.0*pi/Ly)*k

                E[i] += abs(sol.u[i][m,n])^2/(cx^2 + cy^2)
                Z[i] += abs(sol.u[i][m,n])^2

            end
        end
    end

    Plots.plot(sol.t,E,linewidth=2,legend=true,xaxis="t",label="E (Energy)")
    pez = Plots.plot!(sol.t,Z,linewidth=2,legend=true,xaxis="t",label="Z (Enstrophy)",yaxis="E, Z")

    Plots.display(pez)

end

# global code
opt_eqs()

lx = 2.0*Float64(pi)
ly = 2.0*Float64(pi)
nx = 6
ny = 6

@time sol = exec(lx,ly,nx,ny)

u0 = randn(ComplexF64,nx,(2*ny-1))
tspan = (0.0,100.0)
Cp,Cm = nl_coeffs(lx,ly,nx,ny)
p = [nx,ny,Cp,Cm]

du = similar(u0)
@time nl_eqs!(du,u0,p,tspan)
@code_warntype nl_eqs!(du,u0,p,tspan)

Plots.plot(sol,vars=(0,1),linewidth=2,label="(0,-1)",legend=true)
Plots.plot!(sol,vars=(0,2),linewidth=2,label="(0,0)")
Plots.plot!(sol,vars=(0,3),linewidth=2,label="(0,1)")
Plots.plot!(sol,vars=(0,4),linewidth=2,label="(1,-1)")
Plots.plot!(sol,vars=(0,5),linewidth=2,label="(1,0)")
Plots.plot!(sol,vars=(0,6),linewidth=2,label="(1,1)")

energy(lx,ly,nx,ny,sol)
