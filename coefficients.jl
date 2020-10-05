function acoeffs(ly::Float64,ny::Int)

    A = zeros(ComplexF64,2*ny-1)
    return A

end

function acoeffs(ly::Float64,ny::Int,Ω::Float64,Δθ::Float64,τ::Float64)

    A = zeros(ComplexF64,2*ny-1)
    ζjet = zeros(Float64,2*ny-1)

    # jet vorticity is fraction of planetary vorticity
    Ξ::Float64 = 0.6*Ω
    κ::Float64 = τ == 0.0 ? 0.0 : 1.0/τ

    for y in 1:1:2*ny-1
        ζjet[y] = -κ*Ξ*tanh((ly/2.0 - 0.5*(2*y-1)/(2*ny-1)*ly)/Δθ)
    end
    ζjet_fourier = fftshift(fft(ζjet))

    for y in 1:1:2*ny-1
        A[y,1] = ζjet_fourier[y]
    end

    return A

end

function bcoeffs(lx::Float64,ly::Float64,nx::Int,ny::Int)

    B = zeros(ComplexF64,2*ny-1,nx)
    return B

end

function bcoeffs(lx::Float64,ly::Float64,nx::Int,ny::Int,νn::Float64)

    B = zeros(ComplexF64,2*ny-1,nx)

    twopi::Float64 = 2.0*Float64(pi)

    # hyperviscosity normalized to result in unity dissipation rate at kmax
    α::Int = 2
    kxmax::Float64 = 2.0*Float64(pi)/lx*Float64(nx-1)
    kymax::Float64 = 2.0*Float64(pi)/ly*Float64(ny-1)

    for m = 0:1:nx-1
        nmin = m == 0 ? 1 : -(ny-1)
        for n=nmin:1:ny-1

            kx::Float64 = twopi*Float64(m)/lx
            ky::Float64 = twopi*Float64(n)/ly

            B[n+ny,m+1] = - νn*((kx^2 + ky^2)/(kxmax^2 + kymax^2))^(2*α)

        end
    end

    return B

end

function bcoeffs(lx::Float64,ly::Float64,nx::Int,ny::Int,Ω::Float64,θ::Float64)

    B = zeros(ComplexF64,2*ny-1,nx)

    β::Float64 = 2.0*Ω*cos(θ)

    α::Int = 2
    kxmax::Float64 = 2.0*Float64(pi)/lx*Float64(nx-1)
    kymax::Float64 = 2.0*Float64(pi)/ly*Float64(ny-1)

    for m = 0:1:nx-1
        nmin = m == 0 ? 1 : -(ny-1)
        for n=nmin:1:ny-1

            kx::Float64 = 2.0*Float64(pi)*Float64(m)/lx
            ky::Float64 = 2.0*Float64(pi)*Float64(n)/ly

            B[n+ny,m+1] = im*β*kx/(kx^2 + ky^2)

        end
    end

    return B

end

function bcoeffs(lx::Float64,ly::Float64,nx::Int,ny::Int,Ω::Float64,θ::Float64,νn::Float64)

    B = zeros(ComplexF64,2*ny-1,nx)

    β::Float64 = 2.0*Ω*cos(θ)

    # hyperviscosity normalized to result in unity dissipation rate at kmax
    α::Int = 2
    kxmax::Float64 = 2.0*Float64(pi)/lx*Float64(nx-1)
    kymax::Float64 = 2.0*Float64(pi)/ly*Float64(ny-1)

    for m = 0:1:nx-1
        nmin = m == 0 ? 1 : -(ny-1)
        for n=nmin:1:ny-1

            kx::Float64 = 2.0*Float64(pi)*Float64(m)/lx
            ky::Float64 = 2.0*Float64(pi)*Float64(n)/ly

            B[n+ny,m+1] = im*β*kx/(kx^2 + ky^2) - νn*((kx^2 + ky^2)/(kxmax^2 + kymax^2))^(2*α)

        end
    end

    return B

end

function bcoeffs(lx::Float64,ly::Float64,nx::Int,ny::Int,Ω::Float64,θ::Float64,νn::Float64,τ::Float64)

    B = zeros(ComplexF64,2*ny-1,nx)

    κ::Float64 = τ == 0.0 ? 0.0 : 1.0/τ
    β::Float64 = 2.0*Ω*cos(θ)

    # hyperviscosity normalized to result in unity dissipation rate at kmax
    α::Int = 2
    kxmax::Float64 = 2.0*Float64(pi)/lx*Float64(nx-1)
    kymax::Float64 = 2.0*Float64(pi)/ly*Float64(ny-1)

    for m = 0:1:nx-1
        nmin = m == 0 ? 1 : -(ny-1)
        for n=nmin:1:ny-1

            kx::Float64 = 2.0*Float64(pi)*Float64(m)/lx
            ky::Float64 = 2.0*Float64(pi)*Float64(n)/ly

            B[n+ny,m+1] = - κ + im*β*kx/(kx^2 + ky^2) - νn*((kx^2 + ky^2)/(kxmax^2 + kymax^2))^(2*α)

        end
    end

    return B

end

function ccoeffs(lx::Float64,ly::Float64,nx::Int,ny::Int)

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

    return Cp, Cm

end

function ccoeffs(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int)

    M::Int = nx - 1
    N::Int = ny - 1

    Cp = zeros(Float64,2*ny-1,nx,2*ny-1,nx)
    Cm = zeros(Float64,2*ny-1,nx,2*ny-1,nx)

    # Δp = []
    # Δm = []

    # L + L = L
    for m1=0:1:Λ
        n1min = m1 == 0 ? 1 : -N
        for n1=n1min:1:N
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
    for m1=0:1:Λ
        n1min = m1 == 0 ? 1 : -N
        for n1=n1min:1:N

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

    # @show Δp
    # @show Δm
    return Cp,Cm

end
