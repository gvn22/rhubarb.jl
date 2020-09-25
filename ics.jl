function ic_eqm(lx::Float64,ly::Float64,nx::Int,ny::Int,A::Array{ComplexF64,1})

    ζ0 = zeros(ComplexF64,2*ny-1,nx)

    for y in 1:1:2*ny-1
        ζ0[y,1] = A[y]
    end

    return ζ0

end

function ic_eqm(lx::Float64,ly::Float64,nx::Int,ny::Int,Ω::Float64,ν::Float64,τ::Float64)

    Ξ = 0.6*Ω
    Δθ = 0.05
    θ = Float64(pi)/6.0
    β = 2.0*Ω*cos(θ)

    ujet = c_coeffs(lx,ly,nx,ny,Ξ,Δθ)
    ω,v,v4 = l_coeffs(lx,ly,nx,ny,β,ν)

    ζ0 = zeros(ComplexF64,2*ny-1,nx)
    for n=1:1:ny-1

        ζ0[n+ny,1] = -(ujet[n+ny,1]/τ)/(im*ω[n+ny,1] + v[n+ny,1])

    end

    return ζ0
end

function ic_rand(lx::Float64,ly::Float64,nx::Int,ny::Int)

    umn = zeros(ComplexF64,2*ny-1,2*nx-1)

    uxy = randn(Float64,2*ny-1,2*nx-1)
    umn = fftshift(fft(uxy))
    umn[ny,nx] = 0.0 + im*0.0

    return umn[:,nx:2*nx-1]/100.0

end

function ic_cumulants(nx::Int,ny::Int,Λ::Int,u0::Array{ComplexF64,2})

    u0_low = u0[:,1:Λ+1]
    for n = 1:1:ny-1
        u0_low[n,1] = u0_low[2*ny - n,1]
    end
    u0_low[ny,1] = 0.0 + im*0.0

    u0_high = zeros(ComplexF64,2*ny-1,nx-Λ-1,2*ny-1,nx-Λ-1)
    for m1=Λ+1:1:nx-1
        for n1=-ny+1:1:ny-1
            for m2=Λ+1:1:nx-1
                for n2=-ny+1:1:ny-1

                    u0_high[n2+ny,m2-Λ,n1+ny,m1-Λ] = u0[n2+ny,m2+1]*conj(u0[n1+ny,m1+1])

                end
            end
        end
    end

    return ArrayPartition(u0_low,u0_high)

end
