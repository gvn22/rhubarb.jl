using DifferentialEquations
using TimerOutputs,BenchmarkTools
using Plots; plotly()

function nl_coeffs(lx::Float64,ly::Float64,nx::Int,ny::Int)

    M::Int = nx - 1
    N::Int = ny - 1

    Cp = zeros(Float64,2*ny-1,nx,2*ny-1,nx)
    Cm = zeros(Float64,2*ny-1,nx,2*ny-1,nx)

    # Δp = []
    # Δm = []

    # ++ interactions note: +0 has only (0,+n)
    for m1=1:1:M
        for n1=-N:1:N
            for m2=0:1:min(m1,M-m1)

                n2min = m2 == 0 ? 1 : -N
                for n2=max(n2min,-N-n1):1:min(N,N-n1)

                    px::Float64 = 2.0*Float64(pi)/lx*Float64(m1)
                    py::Float64 = 2.0*Float64(pi)/ly*Float64(n1)
                    qx::Float64 = 2.0*Float64(pi)/lx*Float64(m2)
                    qy::Float64 = 2.0*Float64(pi)/ly*Float64(n2)

                    if m1 == m2
                        Cp[n2+ny,m2+1,n1+ny,m1+1] = -(px*qy - qx*py)/(px^2 + py^2)
                    else
                        Cp[n2+ny,m2+1,n1+ny,m1+1] = -(px*qy - qx*py)*(1.0/(px^2 + py^2) - 1.0/(qx^2 + qy^2))
                    end

                    # m::Int = m1 + m2
                    # n::Int = n1 + n2
                    # push!(Δp,[m,n,m1,n1,m2,n2,Cp[n2+ny,m2+1,n1+ny,m1+1]])

                end
            end
        end
    end

    # +- interactions note: - includes (0,-n) because it is conj(0,n)
    for m1=1:1:M
        for n1=-N:1:N
            for m2=0:1:m1

                n2min = m2 == 0 ? 1 : -N
                n2max = m2 == m1 ? n1 - 1 : N
                for n2=max(n2min,n1-N):1:min(n2max,n1+N)

                    px::Float64 = 2.0*Float64(pi)/lx*Float64(m1)
                    py::Float64 = 2.0*Float64(pi)/ly*Float64(n1)
                    qx::Float64 = 2.0*Float64(pi)/lx*Float64(m2)
                    qy::Float64 = 2.0*Float64(pi)/ly*Float64(n2)

                    Cm[n2+ny,m2+1,n1+ny,m1+1] = (px*qy - qx*py)*(1.0/(px^2 + py^2) - 1.0/(qx^2 + qy^2))

                    # m::Int = m1 - m2
                    # n::Int = n1 - n2
                    # push!(Δm,[m,n,m1,n1,-m2,-n2,Cm[n2+ny,m2+1,n1+ny,m1+1]])

                end
            end
        end
    end

    return Cp,Cm

end

function nl_eqs!(du,u,p,t)

    nx::Int,ny::Int,Cp::Array{Float64,4},Cm::Array{Float64,4} = p
    M::Int = nx - 1
    N::Int = ny - 1

    dζ = fill!(similar(du),0)

    # ++ interactions
    for m1=1:1:M
        for n1=-N:1:N
            for m2=0:1:min(m1,M-m1)

                n2min = m2 == 0 ? 1 : -N
                for n2=max(n2min,-N-n1):1:min(N,N-n1)

                    m::Int = m1 + m2
                    n::Int = n1 + n2

                    dζ[n+ny,m+1] += Cp[n2+ny,m2+1,n1+ny,m1+1]*u[n1+ny,m1+1]*u[n2+ny,m2+1]

                end
            end
        end
    end

    # +- interactions
    for m1=1:1:M
        for n1=-N:1:N
            for m2=0:1:m1

                n2min = m2 == 0 ? 1 : -N
                n2max = m2 == m1 ? n1 - 1 : N
                for n2=max(n2min,n1-N):1:min(n2max,n1+N)

                    m::Int = m1 - m2
                    n::Int = n1 - n2

                    dζ[n+ny,m+1] += Cm[n2+ny,m2+1,n1+ny,m1+1]*u[n1+ny,m1+1]*conj(u[n2+ny,m2+1])

                end
            end
        end
    end

    du .= dζ

end

function gql_coeffs(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int)

    M::Int = nx - 1
    N::Int = ny - 1

    Cp = zeros(Float64,2*ny-1,nx,2*ny-1,nx)
    Cm = zeros(Float64,2*ny-1,nx,2*ny-1,nx)

    # Δp = []
    # Δm = []

    # L + L = L
    for m1=1:1:Λ
        for n1=-N:1:N
            for m2=0:1:min(m1,Λ-m1)

                n2min = m2 == 0 ? 1 : -N
                for n2=max(n2min,-N-n1):1:min(N,N-n1)

                    px::Float64 = 2.0*Float64(pi)/lx*Float64(m1)
                    py::Float64 = 2.0*Float64(pi)/ly*Float64(n1)
                    qx::Float64 = 2.0*Float64(pi)/lx*Float64(m2)
                    qy::Float64 = 2.0*Float64(pi)/ly*Float64(n2)

                    if m1 == m2
                        Cp[n2+ny,m2+1,n1+ny,m1+1] = -(px*qy - qx*py)/(px^2 + py^2)
                    else
                        Cp[n2+ny,m2+1,n1+ny,m1+1] = -(px*qy - qx*py)*(1.0/(px^2 + py^2) - 1.0/(qx^2 + qy^2))
                    end

                    # m::Int = m1 + m2
                    # n::Int = n1 + n2
                    # push!(Δp,[m,n,m1,n1,m2,n2,Cp[n2+ny,m2+1,n1+ny,m1+1]])

                end
            end
        end
    end

    # L - L = L
    # note: -L should always include (0,-n)
    for m1=1:1:Λ
        for n1=-N:1:N
            for m2=0:1:m1

                n2min = m2 == 0 ? 1 : -N
                n2max = m2 == m1 ? n1 - 1 : N
                for n2=max(n2min,n1-N):1:min(n2max,n1+N)

                    px::Float64 = 2.0*Float64(pi)/lx*Float64(m1)
                    py::Float64 = 2.0*Float64(pi)/ly*Float64(n1)
                    qx::Float64 = 2.0*Float64(pi)/lx*Float64(m2)
                    qy::Float64 = 2.0*Float64(pi)/ly*Float64(n2)

                    Cm[n2+ny,m2+1,n1+ny,m1+1] = (px*qy - qx*py)*(1.0/(px^2 + py^2) - 1.0/(qx^2 + qy^2))

                    # m::Int = m1 - m2
                    # n::Int = n1 - n2
                    # push!(Δm,[m,n,m1,n1,-m2,-n2,Cm[n2+ny,m2+1,n1+ny,m1+1]])

                end
            end
        end
    end

    # H - H = L
    for m1=Λ+1:1:M
        for n1=-N:1:N
            for m2=max(Λ+1,m1-Λ):1:m1

                n2max = m2 == m1 ? n1 - 1 : N
                for n2=max(-N,n1-N):1:min(n2max,n1+N)

                    px::Float64 = 2.0*Float64(pi)/lx*Float64(m1)
                    py::Float64 = 2.0*Float64(pi)/ly*Float64(n1)
                    qx::Float64 = 2.0*Float64(pi)/lx*Float64(m2)
                    qy::Float64 = 2.0*Float64(pi)/ly*Float64(n2)

                    Cm[n2+ny,m2+1,n1+ny,m1+1] = (px*qy - qx*py)*(1.0/(px^2 + py^2) - 1.0/(qx^2 + qy^2))

                    # m::Int = m1 - m2
                    # n::Int = n1 - n2
                    # push!(Δm,[m,n,m1,n1,-m2,-n2,Cm[n2+ny,m2+1,n1+ny,m1+1]])

                end
            end
        end
    end

    # H + L = H
    for m1=Λ+1:1:M
        for n1=-N:1:N
            for m2=0:1:min(M-m1,Λ)

                n2min = m2 == 0 ? 1 : -N
                for n2=max(n2min,-N-n1):1:min(N,N-n1)

                    px::Float64 = 2.0*Float64(pi)/lx*Float64(m1)
                    py::Float64 = 2.0*Float64(pi)/ly*Float64(n1)
                    qx::Float64 = 2.0*Float64(pi)/lx*Float64(m2)
                    qy::Float64 = 2.0*Float64(pi)/ly*Float64(n2)

                    Cp[n2+ny,m2+1,n1+ny,m1+1] = -(px*qy - qx*py)*(1.0/(px^2 + py^2) - 1.0/(qx^2 + qy^2))

                    # m::Int = m1 + m2
                    # n::Int = n1 + n2
                    # push!(Δp,[m,n,m1,n1,m2,n2,Cp[n2+ny,m2+1,n1+ny,m1+1]])

                end
            end
        end
    end

    # H - L = H
    # note: -L should always include (0,-n)
    for m1=Λ+1:1:M
        for n1=-N:1:N
            for m2=0:1:min(Λ,m1 - Λ - 1)

                n2min = m2 == 0 ? 1 : -N
                for n2=max(n2min,n1-N):1:min(N,n1+N)

                    px::Float64 = 2.0*Float64(pi)/lx*Float64(m1)
                    py::Float64 = 2.0*Float64(pi)/ly*Float64(n1)
                    qx::Float64 = 2.0*Float64(pi)/lx*Float64(m2)
                    qy::Float64 = 2.0*Float64(pi)/ly*Float64(n2)

                    Cm[n2+ny,m2+1,n1+ny,m1+1] = (px*qy - qx*py)*(1.0/(px^2 + py^2) - 1.0/(qx^2 + qy^2))

                    # m::Int = m1 - m2
                    # n::Int = n1 - n2
                    # push!(Δm,[m,n,m1,n1,-m2,-n2,Cm[n2+ny,m2+1,n1+ny,m1+1]])

                end
            end
        end
    end

    return Cp,Cm

end

function gql_eqs!(du,u,p,t)

    nx::Int,ny::Int,Λ::Int,Cp::Array{Float64,4},Cm::Array{Float64,4} = p
    
    M::Int = nx - 1
    N::Int = ny - 1

    dζ = fill!(similar(du),0)

    # L + L = L
    for m1=1:1:Λ
        for n1=-N:1:N
            for m2=0:1:min(m1,Λ-m1)

                n2min = m2 == 0 ? 1 : -N
                for n2=max(n2min,-N-n1):1:min(N,N-n1)

                    m::Int = m1 + m2
                    n::Int = n1 + n2

                    dζ[n+ny,m+1] += Cp[n2+ny,m2+1,n1+ny,m1+1]*u[n1+ny,m1+1]*u[n2+ny,m2+1]

                end
            end
        end
    end

    # L - L = L
    for m1=1:1:Λ
        for n1=-N:1:N
            for m2=0:1:m1

                n2min = m2 == 0 ? 1 : -N
                n2max = m2 == m1 ? n1 - 1 : N
                for n2=max(n2min,n1-N):1:min(n2max,n1+N)

                    m::Int = m1 - m2
                    n::Int = n1 - n2

                    dζ[n+ny,m+1] += Cm[n2+ny,m2+1,n1+ny,m1+1]*u[n1+ny,m1+1]*conj(u[n2+ny,m2+1])

                end
            end
        end
    end

    # H - H = L
    for m1=Λ+1:1:M
        for n1=-N:1:N
            for m2=max(Λ+1,m1-Λ):1:m1

                n2max = m2 == m1 ? n1 - 1 : N
                for n2=max(-N,n1-N):1:min(n2max,n1+N)

                    m::Int = m1 - m2
                    n::Int = n1 - n2

                    dζ[n+ny,m+1] += Cm[n2+ny,m2+1,n1+ny,m1+1]*u[n1+ny,m1+1]*conj(u[n2+ny,m2+1])

                end
            end
        end
    end

    # H + L = H
    for m1=Λ+1:1:M
        for n1=-N:1:N
            for m2=0:1:min(M-m1,Λ)

                n2min = m2 == 0 ? 1 : -N
                for n2=max(n2min,-N-n1):1:min(N,N-n1)

                    m::Int = m1 + m2
                    n::Int = n1 + n2

                    dζ[n+ny,m+1] += Cp[n2+ny,m2+1,n1+ny,m1+1]*u[n1+ny,m1+1]*u[n2+ny,m2+1]

                end
            end
        end
    end

    # H - L = H
    for m1=Λ+1:1:M
        for n1=-N:1:N
            for m2=0:1:min(Λ,m1 - Λ - 1)

                n2min = m2 == 0 ? 1 : -N
                for n2=max(n2min,n1-N):1:min(N,n1+N)

                    m::Int = m1 - m2
                    n::Int = n1 - n2

                    dζ[n+ny,m+1] += Cm[n2+ny,m2+1,n1+ny,m1+1]*u[n1+ny,m1+1]*conj(u[n2+ny,m2+1])

                end
            end
        end
    end

    du .= dζ

end


function opt_eqs()

    samples = 7
    timings = zeros(samples)
    for i in 1:1:samples

        nx = i + 1
        ny = i + 1

        println("Solving Nx2N system with N = ", nx)
        u0 = randn(ComplexF64,2*ny-1,nx)
        tspan = (0.0,100.0)
        Cp,Cm = nl_coeffs(lx,ly,nx,ny)
        p = [nx,ny,Cp,Cm]
        prob = ODEProblem(nl_eqs!,u0,tspan,p)
        timings[i] = @elapsed solve(prob,RK4(),adaptive=true,progress=true,save_start=false,save_everystep=false)

    end

    dims = [i + 1 for i in 1:samples]
    Plots.plot(dims,timings,scale=:log,xaxis="N",yaxis="T",markershape = :square,legend=false)

end

function rhu(lx::Float64,ly::Float64,nx::Int,ny::Int)

    u0 = rand(ComplexF64,2*ny-1,nx)
    tspan = (0.0,100.0)
    Cp,Cm = nl_coeffs(lx,ly,nx,ny)
    p = [nx,ny,Cp,Cm]
    prob = ODEProblem(nl_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=100,saveat=10,save_start=false,save_everystep=false)

    return sol
end

function barb(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int)

    u0 = rand(ComplexF64,2*ny-1,nx)
    tspan = (0.0,100.0)
    Cp,Cm = gql_coeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,Cp,Cm]
    prob = ODEProblem(gql_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=100,saveat=10,save_start=false,save_everystep=false)

    return sol
end

function energy(lx,ly,nx,ny,sol)

    E = zeros(Float64,length(sol.u))
    Z = fill!(similar(E),0)

    for i in eachindex(sol.u)

        for m1 = 0:1:nx-1
            n1min = m1 == 0 ? 1 : -ny + 1
            for n1 = n1min:1:ny-1

                cx,cy = (2.0*pi/lx)*m1,(2.0*pi/ly)*n1

                E[i] += abs(sol.u[i][n1 + ny,m1 + 1])^2/(cx^2 + cy^2)
                Z[i] += abs(sol.u[i][n1 + ny,m1 + 1])^2

            end
        end
    end

    Plots.plot(sol.t,E,linewidth=2,legend=true,xaxis="t",label="E (Energy)")
    pez = Plots.plot!(sol.t,Z,linewidth=2,legend=true,xaxis="t",label="Z (Enstrophy)",yaxis="E, Z")

    Plots.display(pez)

    return E,Z

end

# global code
lx = 2.0*Float64(pi)
ly = 2.0*Float64(pi)
nx = 4
ny = 4
Λ = 0

# Δ1,Δ2 = nl_coeffs(lx,ly,nx,ny)
# Γ1,Γ2 = gql_coeffs(lx,ly,nx,ny,Λ)

sol = rhu(lx,ly,nx,ny)
sol = barb(lx,ly,nx,ny,Λ)

E,Z = energy(lx,ly,nx,ny,sol)

Plots.plot(sol)
# Plots.plot(sol,vars=(0,1),linewidth=2,label="(0,-1)",legend=true)
# Plots.plot!(sol,vars=(0,2),linewidth=2,label="(0,0)")
# Plots.plot!(sol,vars=(0,3),linewidth=2,label="(0,1)")
# Plots.plot!(sol,vars=(0,4),linewidth=2,label="(1,-1)")
# Plots.plot!(sol,vars=(0,5),linewidth=2,label="(1,0)")
# Plots.plot!(sol,vars=(0,6),linewidth=2,label="(1,1)")

opt_eqs()

# tests
u0 = randn(ComplexF64,2*ny-1,nx)
tspan = (0.0,1000.0)
Cp,Cm = nl_coeffs(lx,ly,nx,ny)
p = [nx,ny,Cp,Cm]

@btime nl_coeffs(lx,ly,nx,ny)
du = similar(u0)
@time nl_eqs!(du,u0,p,tspan)
@code_warntype nl_eqs!(du,u0,p,tspan)
@code_warntype gql_eqs!(du,u0,p,tspan)
# integrator = init(prob,RK4())
# step!(integrator)
