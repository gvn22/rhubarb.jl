using OrdinaryDiffEq,RecursiveArrayTools,FFTW
using TimerOutputs,BenchmarkTools
using Plots; plotly()

function l_coeffs(lx::Float64,ly::Float64,nx::Int,ny::Int,β::Float64,ν::Float64)

    ω = zeros(Float64,2*ny-1,nx)
    v = zeros(Float64,2*ny-1,nx)
    v4 = zeros(Float64,2*ny-1,nx)

    M::Int = nx - 1
    N::Int = ny - 1

    kxmax::Float64 = 2.0*Float64(pi)/lx*Float64(M)
    kymax::Float64 = 2.0*Float64(pi)/ly*Float64(N)

    α::Int = 2

    for m = 0:1:M
        nmin = m == 0 ? 1 : -N
        for n=nmin:1:N

            kx::Float64 = 2.0*Float64(pi)/lx*Float64(m)
            ky::Float64 = 2.0*Float64(pi)/ly*Float64(n)

            ω[n+ny,m+1] = β*kx/(kx^2 + ky^2)
            v[n+ny,m+1] = -ν*(kx^2 + ky^2)

            v4[n+ny,m+1] = -ν*((kx^2 + ky^2)/(kxmax^2 + kymax^2))^(2*α)

        end
    end

    return ω,v,v4

end

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

    nx::Int,ny::Int,ω::Array{Float64,2},v::Array{Float64,2},Cp::Array{Float64,4},Cm::Array{Float64,4} = p
    M::Int = nx - 1
    N::Int = ny - 1

    dζ = fill!(similar(du),0)

    for m = 0:1:M
        nmin = m == 0 ? 1 : -N
        for n=nmin:1:N

            dζ[n+ny,m+1] += im*ω[n+ny,m+1]*u[n+ny,m+1]
            dζ[n+ny,m+1] += v[n+ny,m+1]*u[n+ny,m+1]

        end
    end

    # ++ interactions
    for m1=1:1:M
        for n1=-N:1:N
            for m2=0:1:min(m1,M-m1)

                n2min = m2 == 0 ? 1 : -N
                for n2=max(n2min,-N-n1):1:min(N,N-n1)

                    m::Int = m1 + m2
                    n::Int = n1 + n2

                    dζ[n+ny,m+1] += Cp[n2+ny,m2+1,n1+ny,m1+1]*u[n1+ny,m1+1]*u[n2+ny,m2+1]
                    # could eliminate addition on array indices

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

    nx::Int,ny::Int,Λ::Int,ω::Array{Float64,2},v::Array{Float64,2},Cp::Array{Float64,4},Cm::Array{Float64,4} = p

    M::Int = nx - 1
    N::Int = ny - 1

    dζ = fill!(similar(du),0)

    # linear terms
    for m = 0:1:M
        nmin = m == 0 ? 1 : -N
        for n=nmin:1:N

            dζ[n+ny,m+1] += im*ω[n+ny,m+1]*u[n+ny,m+1]
            dζ[n+ny,m+1] += v[n+ny,m+1]*u[n+ny,m+1]

        end
    end

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

function gce2_eqs!(du,u,p,t)

    nx::Int,ny::Int,Λ::Int,ω::Array{Float64,2},v::Array{Float64,2},Cp::Array{Float64,4},Cm::Array{Float64,4} = p

    M::Int = nx - 1
    N::Int = ny - 1

    dζ = fill!(similar(du.x[1]),0)
    dΘ = fill!(similar(du.x[2]),0)

    temp_li = fill!(similar(du.x[2]),0)
    temp_nl = fill!(similar(du.x[2]),0)

    # linear terms: L
    for m = 0:1:Λ
        nmin = m == 0 ? 1 : -N
        for n=nmin:1:N

            dζ[n+ny,m+1] += im*ω[n+ny,m+1]*u.x[1][n+ny,m+1]
            dζ[n+ny,m+1] += v[n+ny,m+1]*u.x[1][n+ny,m+1]

        end
    end

    # L + L = L
    # println("L+L = L")
    for m1=1:1:Λ
        for n1=-N:1:N
            for m2=0:1:min(m1,Λ-m1)

                n2min = m2 == 0 ? 1 : -N
                for n2=max(n2min,-N-n1):1:min(N,N-n1)

                    m::Int = m1 + m2
                    n::Int = n1 + n2

                    # @show m,n,m1,n1,m2,n2

                    dζ[n+ny,m+1] += Cp[n2+ny,m2+1,n1+ny,m1+1]*u.x[1][n1+ny,m1+1]*u.x[1][n2+ny,m2+1]

                end
            end
        end
    end

    # L - L = L
    # println("L-L = L")
    for m1=1:1:Λ
        for n1=-N:1:N
            for m2=0:1:m1

                n2min = m2 == 0 ? 1 : -N
                n2max = m2 == m1 ? n1 - 1 : N
                for n2=max(n2min,n1-N):1:min(n2max,n1+N)

                    m::Int = m1 - m2
                    n::Int = n1 - n2

                    # @show m,n,m1,n1,-m2,-n2
                    dζ[n+ny,m+1] += Cm[n2+ny,m2+1,n1+ny,m1+1]*u.x[1][n1+ny,m1+1]*conj(u.x[1][n2+ny,m2+1])

                end
            end
        end
    end

    # H - H = L
    # println("H-H = L")
    for m1=Λ+1:1:M
        for n1=-N:1:N
            for m2=max(Λ+1,m1-Λ):1:m1

                n2max = m2 == m1 ? n1 - 1 : N
                for n2=max(-N,n1-N):1:min(n2max,n1+N)

                    m::Int = m1 - m2
                    n::Int = n1 - n2

                    # @show m,n,m1,n1,-m2,-n2

                    # note: u.x[2] contains H2*conj(H1) so H-H is conj(H2)*H1
                    dζ[n+ny,m+1] += Cm[n2+ny,m2+1,n1+ny,m1+1]*conj(u.x[2][n2+ny,m2-Λ,n1+ny,m1-Λ])

                end
            end
        end
    end

    # linear terms: H
    for m = Λ+1:1:M
        for n=-N:1:N

            temp_li[n+ny,m-Λ,n+ny,m-Λ] += im*ω[n+ny,m+1]
            temp_li[n+ny,m-Λ,n+ny,m-Λ] += v[n+ny,m+1]

        end
    end

    # H + L = H
    # println("H+L = H")
    for m1=Λ+1:1:M
        for n1=-N:1:N
            for m2=0:1:min(M-m1,Λ)

                n2min = m2 == 0 ? 1 : -N
                for n2=max(n2min,-N-n1):1:min(N,N-n1)

                    m::Int = m1 + m2
                    n::Int = n1 + n2

                    c = Cp[n2+ny,m2+1,n1+ny,m1+1]
                    # @show m,n,m1,n1,m2,n2,c

                    temp_nl[n1+ny,m1-Λ,n+ny,m-Λ] += Cp[n2+ny,m2+1,n1+ny,m1+1]*u.x[1][n2+ny,m2+1]

                end
            end
        end
    end

    # H - L = H
    # println("H-L = H")
    for m1=Λ+1:1:M
        for n1=-N:1:N
            for m2=0:1:min(Λ,m1 - Λ - 1)

                n2min = m2 == 0 ? 1 : -N
                for n2=max(n2min,n1-N):1:min(N,n1+N)

                    m::Int = m1 - m2
                    n::Int = n1 - n2

                    c = Cm[n2+ny,m2+1,n1+ny,m1+1]
                    # @show m,n,m1,n1,-m2,-n2,c

                    temp_nl[n1+ny,m1-Λ,n+ny,m-Λ] += Cm[n2+ny,m2+1,n1+ny,m1+1]*conj(u.x[1][n2+ny,m2+1])

                end
            end
        end
    end

    # H'*H
    # println("HH+")
    for m3=Λ+1:1:M
        for n3=-N:1:N
            for m=Λ+1:1:M
                for n=-N:1:N

                    accumulator_nl::ComplexF64 = 0.0 + im*0.0
                    accumulator_li::ComplexF64 = 0.0 + im*0.0

                    # from H+L
                    for m1=max(Λ+1,m-Λ):1:min(M,m)
                        n2min = m1 == m ? 1 : -N
                        for n1=max(-N,n-N):1:min(n-n2min,N)

                            # d = temp_nl[n1+ny,m1-Λ,n+ny,m-Λ]
                            # @show m3,n3,m,n,m1,n1,d
                            accumulator_nl += temp_nl[n1+ny,m1-Λ,n+ny,m-Λ]*u.x[2][n1+ny,m1-Λ,n3+ny,m3-Λ]
                            # accumulator_li += temp_li[n1+ny,m1-Λ,n+ny,m-Λ]*u.x[2][n1+ny,m1-Λ,n3+ny,m3-Λ]

                        end
                    end

                    # from H-L
                    for m1=max(Λ+1,m):1:min(M,m+Λ)
                        n2max = m1 == m ? -1 : N
                        for n1=max(-N,n-n2max):1:min(n+N,N)

                            # d = temp_nl[n1+ny,m1-Λ,n+ny,m-Λ]
                            # @show m3,n3,m,n,m1,n1,d
                            accumulator_nl += temp_nl[n1+ny,m1-Λ,n+ny,m-Λ]*u.x[2][n1+ny,m1-Λ,n3+ny,m3-Λ]
                            # accumulator_li += temp_li[n1+ny,m1-Λ,n+ny,m-Λ]*u.x[2][n1+ny,m1-Λ,n3+ny,m3-Λ]

                        end
                    end

                    dΘ[n+ny,m-Λ,n3+ny,m3-Λ] += accumulator_nl + accumulator_li

                end
            end
        end
    end

    du.x[1] .= dζ
    for m3=Λ+1:1:M
        for n3=-N:1:N
            for m=Λ+1:1:M
                for n=-N:1:N

                    du.x[2][n+ny,m-Λ,n3+ny,m3-Λ] = dΘ[n+ny,m-Λ,n3+ny,m3-Λ] + conj(dΘ[n3+ny,m3-Λ,n+ny,m-Λ])

                end
            end
        end
    end
    # du.x[2] .= dΘ

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

function exec(lx::Float64,ly::Float64,nx::Int,ny::Int,β::Float64,ν::Float64,u0::Array{ComplexF64,2})

    # u0 = rand(ComplexF64,2*ny-1,nx)
    # u0 = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    tspan = (0.0,200.0)
    ω,v,v4 = l_coeffs(lx,ly,nx,ny,β,ν)
    Cp,Cm = nl_coeffs(lx,ly,nx,ny)
    p = [nx,ny,ω,v4,Cp,Cm]
    prob = ODEProblem(nl_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=50)

    return sol
end

function exec(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,β::Float64,ν::Float64,u0::Array{ComplexF64,2})

    # u0 = rand(ComplexF64,2*ny-1,nx)
    # u0 = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    tspan = (0.0,200.0)
    ω,v,v4 = l_coeffs(lx,ly,nx,ny,β,ν)
    Cp,Cm = gql_coeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,ω,v4,Cp,Cm]
    prob = ODEProblem(gql_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=50)

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

function energy(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,sol)

    E = zeros(Float64,length(sol.u))
    Z = fill!(similar(E),0)

    for i in eachindex(sol.u)

        for m1 = 0:1:Λ
            n1min = m1 == 0 ? 1 : -ny + 1
            for n1 = n1min:1:ny-1

                cx,cy = (2.0*pi/lx)*m1,(2.0*pi/ly)*n1

                E[i] += abs(sol.u[i].x[1][n1 + ny,m1 + 1])^2/(cx^2 + cy^2)
                Z[i] += abs(sol.u[i].x[1][n1 + ny,m1 + 1])^2

            end
        end

        for m1 = Λ+1:1:nx-1
            for n1 = -ny+1:1:ny-1

                cx,cy = (2.0*pi/lx)*m1,(2.0*pi/ly)*n1

                E[i] += abs(sol.u[i].x[2][n1 + ny,m1 - Λ,n1 + ny,m1 - Λ])/(cx^2 + cy^2)
                Z[i] += abs(sol.u[i].x[2][n1 + ny,m1 - Λ,n1 + ny,m1 - Λ])

            end
        end
    end

    Plots.plot(sol.t,E,linewidth=2,legend=true,xaxis="t",label="E (Energy)")
    pez = Plots.plot!(sol.t,Z,linewidth=2,legend=true,xaxis="t",label="Z (Enstrophy)",yaxis="E, Z")

    Plots.display(pez)

    return E,Z

end

function inversefourier(sol,nx::Int,ny::Int)

    umn = zeros(ComplexF64,2*ny-1,2*nx-1,length(sol.u))
    uff = zeros(ComplexF64,2*ny-1,2*nx-1,length(sol.u))
    uxy = zeros(Float64,2*ny-1,2*nx-1,length(sol.u))

    for i in eachindex(sol.u)

        for m1 = 0:1:nx-1
            n1min = m1 == 0 ? 1 : -ny + 1
            for n1 = n1min:1:ny-1

                umn[n1 + ny,m1+nx,i] = sol.u[i][n1+ny,m1+1]
                umn[-n1 + ny,-m1+nx,i] = conj(sol.u[i][n1+ny,m1+1])

                if n1 == 0 && m1 == 0
                    umn[-n1 + ny,-m1+nx,i] = 0.0 + im*0.0
                end

            end
        end

        uff[:,:,i] = ifft(ifftshift(umn[:,:,i]))

        for m=1:1:2*nx-1
            for n=1:1:2*ny-1
                uxy[m,n,i] = real(uff[n,m,i])
            end
        end

    end

    return uxy,uff

end

function plot4time(var,fn::String,lx::Float64,ly::Float64,nx::Int,ny::Int)

    x = LinRange(-lx,lx,2*nx-1)
    y = LinRange(-ly,ly,2*ny-1)

    anim = @animate for i ∈ 1:length(var[1,1,:])
        Plots.plot(x,y,var[:,:,i],st=:contourf,color=:bwr,xaxis="x",yaxis="y",title=(i-1)*50,aspect=:equal)
    end
    gif(anim, fn, fps = 0.5)
    # return nothing
end

# global code
lx = 2.0*Float64(pi)
ly = 2.0*Float64(pi)
nx = 3
ny = 3
x = LinRange(-lx,lx,2*nx-1)
y = LinRange(-ly,ly,2*ny-1)
Λ = 1
Ω = 2.0*Float64(pi)
θ = Float64(pi)/3.0
β = 2.0*Ω*cos(θ)
Ξ = 0.6*Ω
Δθ = 0.05
ν = 0.0

uin = randn(ComplexF64,2*ny-1,nx)

# NL solution
u0 = uin
sol1 = exec(lx,ly,nx,ny,β,ν,u0)
E1,Z1 = energy(lx,ly,nx,ny,sol1)

uxy,umn = inversefourier(sol1,nx,ny)
Plots.plot(x,y,uxy[:,:,begin],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
Plots.plot(x,y,uxy[:,:,end],st=:contourf,color=:bwr,xaxis="x",yaxis="y")

# plot4time(uxy,"zeta_beta.gif",lx,ly,nx,ny)

# GQL solution
u0 = uin
sol2 = exec(lx,ly,nx,ny,Λ,β,ν,u0)
E2,Z2 = energy(lx,ly,nx,ny,sol2)

uxy,umn = inversefourier(sol2,nx,ny)
Plots.plot(x,y,uxy[:,:,begin],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
Plots.plot(x,y,uxy[:,:,end],st=:contourf,color=:bwr,xaxis="x",yaxis="y")

# GCE2 solution
# uin = zeros(ComplexF64,2*ny-1,nx)
# for m1 = 0:1:nx-1
#     for n1 = -ny+1:ny-1
#         uin[n1+ny,m1+1] = (m1+ 1.0)*(n1+ny) + im*0.0
#     end
# end

u0_low = uin[:,1:Λ+1]
u0_high = zeros(ComplexF64,2*ny-1,nx - Λ - 1,2*ny-1,nx - Λ - 1)
for m1=Λ+1:1:nx-1
    for n1=-ny+1:1:ny-1
        for m2=Λ+1:1:nx-1
            for n2=-ny+1:1:ny-1

                u0_high[n2+ny,m2-Λ,n1+ny,m1-Λ] = uin[n2+ny,m2+1]*conj(uin[n1+ny,m1+1])

            end
        end
    end
end
u0 = ArrayPartition(u0_low,u0_high)
tspan = (0.0,200.0)
ω,v,v4 = l_coeffs(lx,ly,nx,ny,β,ν)
Cp,Cm = gql_coeffs(lx,ly,nx,ny,Λ)
p = [nx,ny,Λ,ω,v4,Cp,Cm]
prob = ODEProblem(gce2_eqs!,u0,tspan,p)
@time sol3 = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=10)
# integrator = init(prob,RK4())
# step!(integrator)

E3,Z3 = energy(lx,ly,nx,ny,Λ,sol3)

# compare energy and enstrophy curves
Plots.plot(sol1.t,E1,linewidth=2,label="NL")
Plots.plot!(sol2.t,E2,linewidth=2,label="GQL(1)")
Plots.plot!(sol3.t,E3,linewidth=2,label="GCE2(1)")

# tests
# Plots.plot(sol,vars=(0,1),linewidth=2,label="(0,-1)",legend=true)
# Plots.plot!(sol,vars=(0,2),linewidth=2,label="(0,0)")
# Plots.plot!(sol,vars=(0,3),linewidth=2,label="(0,1)")
# Plots.plot!(sol,vars=(0,4),linewidth=2,label="(1,-1)")
# Plots.plot!(sol,vars=(0,5),linewidth=2,label="(1,0)")
# Plots.plot!(sol,vars=(0,6),linewidth=2,label="(1,1)")

# opt_eqs()

# u0 = randn(ComplexF64,2*ny-1,nx)
# tspan = (0.0,1000.0)
# Cp,Cm = nl_coeffs(lx,ly,nx,ny)
# p = [nx,ny,Cp,Cm]

# @btime nl_coeffs(lx,ly,nx,ny)
du = similar(u0)
# @time nl_eqs!(du,u0,p,tspan)
# @code_warntype nl_eqs!(du,u0,p,tspan)
# @code_warntype gql_eqs!(du,u0,p,tspan)
@code_warntype gce2_eqs!(du,u0,p,tspan)
