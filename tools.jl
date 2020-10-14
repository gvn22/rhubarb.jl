function ispositive(cumulant::Array{ComplexF64,4},nx::Int,ny::Int,Λ::Int)
    twopoint = reshape(cumulant,(2*ny-1)*(nx-Λ),(2*ny-1)*(nx-Λ))
    D = eigvals(twopoint)
    # @info "Is positive? ", !any(x->x<0,D)
    !any(x->x<0.0,D)
end

function positivity!(cumulant::Array{ComplexF64,4},nx::Int,ny::Int,Λ::Int)
    twopoint = reshape(cumulant,(2*ny-1)*(nx-Λ),(2*ny-1)*(nx-Λ))
    D,V = eigen(twopoint)
    @info "Removing following eignvalues from second cumulant ", D[isless.(D,0)]
    Dpos = max.(D,0.0)
    twopoint = V*diagm(Dpos)*inv(V)
    # optimise further:
    # mul!(twopoint,V,lmul!(diagm(Dpos),inv(V)))
    cumulant = reshape(twopoint,2*ny-1,nx-Λ,2*ny-1,nx-Λ)
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

    for i in eachindex(ufft)

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
    uxy
end

function inversefourier(nx::Int,ny::Int,Λ::Int,u::Array{ArrayPartition{Complex{Float64},Tuple{Array{Complex{Float64},2},Array{Complex{Float64},4}}},1})

    umn = zeros(ComplexF64,2*ny-1,2*nx-1,length(u))
    uff = zeros(ComplexF64,2*ny-1,2*nx-1,length(u))
    uxy = zeros(Float64,2*ny-1,2*nx-1,length(u))

    for i in eachindex(u)

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
    uxy
end
