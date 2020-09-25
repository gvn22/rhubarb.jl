# energy for NL/GQL
function energy(lx::Float64,ly::Float64,nx::Int,ny::Int,u::Array{Array{ComplexF64,2},1})

    E = zeros(Float64,length(u))
    Z = fill!(similar(E),0)

    for i in eachindex(u)

        for m1 = 0:1:nx-1
            n1min = m1 == 0 ? 1 : -(ny-1)
            for n1 = n1min:1:ny-1

                kx = 2.0*Float64(pi)/lx*m1
                ky = 2.0*Float64(pi)/ly*n1

                E[i] += abs(u[i][n1 + ny,m1 + 1])^2/(kx^2 + ky^2)
                Z[i] += abs(u[i][n1 + ny,m1 + 1])^2

            end
        end
    end

    return E,Z

end

# energy for GCE2
function energy(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,u::Array{ArrayPartition{Complex{Float64},Tuple{Array{Complex{Float64},2},Array{Complex{Float64},4}}},1})

    E = zeros(Float64,length(u))
    Z = fill!(similar(E),0)

    for i in eachindex(u)

        for m1 = 0:1:Λ
            n1min = m1 == 0 ? 1 : -(ny-1)
            for n1 = n1min:1:ny-1

                kx = 2.0*Float64(pi)/lx*m1
                ky = 2.0*Float64(pi)/ly*n1

                E[i] += abs(u[i].x[1][n1 + ny,m1 + 1])^2/(kx^2 + ky^2)
                Z[i] += abs(u[i].x[1][n1 + ny,m1 + 1])^2

            end
        end

        for m1 = Λ+1:1:nx-1
            for n1 = -(ny-1):1:ny-1

                kx = 2.0*Float64(pi)/lx*m1
                ky = 2.0*Float64(pi)/ly*n1

                E[i] += abs(u[i].x[2][n1 + ny,m1 - Λ,n1 + ny,m1 - Λ])/(kx^2 + ky^2)
                Z[i] += abs(u[i].x[2][n1 + ny,m1 - Λ,n1 + ny,m1 - Λ])

            end
        end
    end

    return E,Z

end

# zonal power in NL/GQL
function zonalpower(sol,lx::Float64,ly::Float64,nx::Int,ny::Int)

    E = zeros(Float64,length(sol.u),nx)
    Z = fill!(similar(E),0)

    for i in eachindex(sol.u)

        for m1 = 0:1:nx-1
            n1min = m1 == 0 ? 1 : -ny + 1
            for n1 = n1min:1:ny-1

                cx,cy = (2.0*pi/lx)*m1,(2.0*pi/ly)*n1

                E[i,m1+1] += abs(sol.u[i][n1 + ny,m1 + 1])^2/(cx^2 + cy^2)
                Z[i,m1+1] += abs(sol.u[i][n1 + ny,m1 + 1])^2

            end
        end

    end

    return E,Z

end

# zonal power in GCE2
function zonalpower(sol,lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int)

    E = zeros(Float64,length(sol.u),nx)
    Z = fill!(similar(E),0)

    for i in eachindex(sol.u)

        for m1 = 0:1:Λ
            n1min = m1 == 0 ? 1 : -ny + 1
            for n1 = n1min:1:ny-1

                cx,cy = (2.0*pi/lx)*m1,(2.0*pi/ly)*n1

                E[i,m1+1] += abs(sol.u[i].x[1][n1 + ny,m1 + 1])^2/(cx^2 + cy^2)
                Z[i,m1+1] += abs(sol.u[i].x[1][n1 + ny,m1 + 1])^2

            end
        end

        for m1 = Λ+1:1:nx-1
            for n1 = -ny+1:1:ny-1

                cx,cy = (2.0*pi/lx)*m1,(2.0*pi/ly)*n1

                E[i,m1+1] += abs(sol.u[i].x[2][n1 + ny,m1 - Λ,n1 + ny,m1 - Λ])/(cx^2 + cy^2)
                Z[i,m1+1] += abs(sol.u[i].x[2][n1 + ny,m1 - Λ,n1 + ny,m1 - Λ])

            end
        end

    end

    return E,Z

end

# mean vorticity NL/GQL
function meanvorticity(sol,lx::Float64,ly::Float64,nx::Int,ny::Int)

    ζf = zeros(ComplexF64,length(sol.u),2*ny-1)
    ζy = zeros(Float64,length(sol.u),2*ny-1)

    for i in eachindex(sol.u)

        for n1 = 1:1:ny-1

            ζf[i,n1+ny] = sol.u[i][n1+ny,1]
            ζf[i,-n1+ny] = conj(sol.u[i][-n1+ny,1])

        end

        ζf[i,ny] = 0.0 + 0.0im
        ζy[i,:] .= real(ifft(ifftshift(ζf[i,:])))

    end

    return ζy

end

# mean vorticity GCE2
function meanvorticity(sol,lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int)

    ζf = zeros(ComplexF64,length(sol.u),2*ny-1)
    ζy = zeros(Float64,length(sol.u),2*ny-1)

    for i in eachindex(sol.u)

        for n1 = 1:1:ny-1

            ζf[i,n1+ny] = sol.u[i].x[1][n1+ny,1]
            ζf[i,-n1+ny] = conj(sol.u[i].x[1][-n1+ny,1])

        end

        ζf[i,ny] = 0.0 + 0.0im
        ζy[i,:] .= real(ifft(ifftshift(ζf[i,:])))

    end

    return ζy

end
