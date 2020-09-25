
function ispositive(cumulant::Array{ComplexF64,4},nx::Int,ny::Int,Λ::Int)

    twopoint = zeros(ComplexF64,(2*ny-1)*nx,(2*ny-1)*nx)

    for m=Λ+1:1:nx-1
        for n=-ny+1:1:ny-1
            for m3=Λ+1:1:nx-1
                for n3=-ny+1:1:ny-1

                    twopoint[(n+ny)*(nx-1) + m-Λ,(n3+ny)*(nx-1) + m3-Λ] = cumulant[n+ny,m-Λ,n3+ny,m3-Λ]

                end
            end
        end
    end

    return isposdef(twopoint)

end

function positivity!(cumulant::Array{ComplexF64,4},nx::Int,ny::Int,Λ::Int)

    println("Removing negative eignvalues from two-point correlation...")
    twopoint = zeros(ComplexF64,(2*ny-1)*nx,(2*ny-1)*nx)

    for m=Λ+1:1:nx-1
        for n=-ny+1:1:ny-1
            for m3=Λ+1:1:nx-1
                for n3=-ny+1:1:ny-1

                    twopoint[(n+ny)*(nx-1) + m-Λ,(n3+ny)*(nx-1) + m3-Λ] = cumulant[n+ny,m-Λ,n3+ny,m3-Λ]

                end
            end
        end
    end

    D = eigvals(twopoint)
    V = eigvecs(twopoint)
    D_pos = [d = real(d) < 0.0 ? 0.0 + imag(d)*im : real(d) + imag(d)*im for d in D]
    # use mul!
    twopoint = V*(D_pos.*V')

    for m=Λ+1:1:nx-1
        for n=-ny+1:1:ny-1
            for m3=Λ+1:1:nx-1
                for n3=-ny+1:1:ny-1

                     cumulant[n+ny,m-Λ,n3+ny,m3-Λ] = twopoint[(n+ny)*(nx-1) + m-Λ,(n3+ny)*(nx-1) + m3-Λ]

                end
            end
        end
    end

end

function inversefourier(nx::Int,ny::Int,uff::Array{ComplexF64,2})

    umn = zeros(ComplexF64,2*ny-1,2*nx-1)
    uxy = zeros(Float64,2*ny-1,2*nx-1)

    for m1 = 0:1:nx-1
        n1min = m1 == 0 ? 1 : -ny + 1
        for n1 = n1min:1:ny-1

            umn[n1 + ny,m1+nx] = uff[n1+ny,m1+1]
            umn[-n1 + ny,-m1+nx] = conj(uff[n1+ny,m1+1])

        end
    end

    umn[ny,nx] = 0.0 + im*0.0

    uxy = real(ifft(ifftshift(umn)))

    return uxy

end

function inversefourier(nx::Int,ny::Int,ufft::Array{Array{ComplexF64,2},1})

    umn = zeros(ComplexF64,2*ny-1,2*nx-1,length(ufft))
    uff = zeros(ComplexF64,2*ny-1,2*nx-1,length(ufft))
    uxy = zeros(Float64,2*ny-1,2*nx-1,length(ufft))

    for i in eachindex(sol.u)

        for m1 = 0:1:nx-1
            n1min = m1 == 0 ? 1 : -ny + 1
            for n1 = n1min:1:ny-1

                umn[n1 + ny,m1+nx,i] = ufft[i][n1+ny,m1+1]
                umn[-n1 + ny,-m1+nx,i] = conj(ufft[i][n1+ny,m1+1])

            end
        end

        umn[ny,nx,i] = 0.0 + im*0.0

        uff[:,:,i] = ifft(ifftshift(umn[:,:,i]))

        for m=1:1:2*nx-1
            for n=1:1:2*ny-1
                uxy[n,m,i] = real(uff[n,m,i])
            end
        end

    end

    return uxy

end

function inversefourier(nx::Int,ny::Int,Λ::Int,u::Array{ArrayPartition{Complex{Float64},Tuple{Array{Complex{Float64},2},Array{Complex{Float64},4}}},1})

    umn = zeros(ComplexF64,2*ny-1,2*nx-1,length(u))
    uff = zeros(ComplexF64,2*ny-1,2*nx-1,length(u))
    uxy = zeros(Float64,2*ny-1,2*nx-1,length(u))

    for i in eachindex(sol.u)

        for m1 = 0:1:Λ
            n1min = m1 == 0 ? 1 : -ny + 1
            for n1 = n1min:1:ny-1

                umn[n1 + ny,m1+nx,i] = u[i].x[1][n1+ny,m1+1]
                umn[-n1 + ny,-m1+nx,i] = conj(u[i].x[1][n1+ny,m1+1])

            end
        end

        umn[ny,nx,i] = 0.0 + im*0.0

        uff[:,:,i] = ifft(ifftshift(umn[:,:,i]))

        for m=1:1:2*nx-1
            for n=1:1:2*ny-1
                uxy[n,m,i] = real(uff[n,m,i])
            end
        end

    end

    return uxy

end

function testfourier(nx::Int,ny::Int)

    u0xy = rand(Float64,2*ny-1,2*nx-1)
    uff = fftshift(fft(u0xy))

    umn = zeros(ComplexF64,2*ny-1,2*nx-1)
    uxy = zeros(Float64,2*ny-1,2*nx-1)
    uxy2 = zeros(Float64,2*ny-1,2*nx-1)

    for m1 = 0:1:nx-1
        n1min = m1 == 0 ? 1 : -ny + 1
        for n1 = n1min:1:ny-1

            umn[n1 + ny,m1+nx] = uff[n1+ny,m1+nx]
            umn[-n1 + ny,-m1+nx] = conj(uff[n1+ny,m1+nx])

        end
    end

    uxy .= real(ifft(ifftshift(umn)))
    uxy2 .= real(ifft(ifftshift(uff)))

    return u0xy,uxy,uxy2

end
