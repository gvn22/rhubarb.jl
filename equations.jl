function nl_eqs!(du,u,p,t)

    nx::Int,ny::Int,A::Array{ComplexF64,1},B::Array{ComplexF64,2},Cp::Array{Float64,4},Cm::Array{Float64,4} = p

    M::Int = nx - 1
    N::Int = ny - 1

    dζ = fill!(similar(du),0)

    for n=1:1:N

        dζ[n+ny,1] += A[n+ny]

    end

    for m = 0:1:M
        nmin = m == 0 ? 1 : -N
        for n=nmin:1:N

            dζ[n+ny,m+1] += B[n+ny,m+1]*u[n+ny,m+1]

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

function gql_eqs!(du,u,p,t)

    nx::Int,ny::Int,Λ::Int,A::Array{ComplexF64,1},B::Array{ComplexF64,2},Cp::Array{Float64,4},Cm::Array{Float64,4} = p

    M::Int = nx - 1
    N::Int = ny - 1

    dζ = fill!(similar(du),0)

    # constant terms
    for n=1:1:N

        dζ[n+ny,1] += A[n+ny]

    end

    # linear terms
    for m = 0:1:M
        nmin = m == 0 ? 1 : -N
        for n=nmin:1:N

            dζ[n+ny,m+1] += B[n+ny,m+1]*u[n+ny,m+1]

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

    nx::Int,ny::Int,Λ::Int,A::Array{ComplexF64,1},B::Array{ComplexF64,2},Cp::Array{Float64,4},Cm::Array{Float64,4} = p

    M::Int = nx - 1
    N::Int = ny - 1

    dζ = fill!(similar(u.x[1]),0)
    dΘ = fill!(similar(u.x[2]),0)

    # constant terms
    for n=1:1:N

        dζ[n+ny,1] += A[n+ny]

    end

    # low mode equations
    # linear terms: L
    for m = 0:1:Λ
        nmin = m == 0 ? 1 : -N
        for n=nmin:1:N

            dζ[n+ny,m+1] += B[n+ny,m+1]*u.x[1][n+ny,m+1]

        end
    end

    # L + L = L
    for m1=1:1:Λ

        n1min = m1 == 0 ? 1 : -N
        for n1=n1min:1:N

            for m2=0:1:min(m1,Λ-m1)

                n2min = m2 == 0 ? 1 : -N
                for n2=max(n2min,-N-n1):1:min(N,N-n1)

                    m::Int = m1 + m2
                    n::Int = n1 + n2

                    dζ[n+ny,m+1] += Cp[n2+ny,m2+1,n1+ny,m1+1]*u.x[1][n1+ny,m1+1]*u.x[1][n2+ny,m2+1]

                end
            end
        end
    end

    # L - L = L
    for m1=1:1:Λ
        n1min = m1 == 0 ? 1 : -N
        for n1=n1min:1:N
            for m2=0:1:m1

                n2min = m2 == 0 ? 1 : -N
                n2max = m2 == m1 ? n1 - 1 : N
                for n2=max(n2min,n1-N):1:min(n2max,n1+N)

                    m::Int = m1 - m2
                    n::Int = n1 - n2

                    dζ[n+ny,m+1] += Cm[n2+ny,m2+1,n1+ny,m1+1]*u.x[1][n1+ny,m1+1]*conj(u.x[1][n2+ny,m2+1])

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

                    # note: u.x[2] contains H2*conj(H1) so H-H is conj(H2)*H1
                    dζ[n+ny,m+1] += Cm[n2+ny,m2+1,n1+ny,m1+1]*conj(u.x[2][n2+ny,m2-Λ,n1+ny,m1-Λ])

                end
            end
        end
    end

    # field bilinear equations
    temp_li = fill!(similar(u.x[2]),0)
    temp_nl = fill!(similar(u.x[2]),0)

    # linear terms: H
    for m = Λ+1:1:M
        for n=-N:1:N

            temp_li[n+ny,m-Λ,n+ny,m-Λ] += B[n+ny,m+1]

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

                    temp_nl[n1+ny,m1-Λ,n+ny,m-Λ] += Cp[n2+ny,m2+1,n1+ny,m1+1]*u.x[1][n2+ny,m2+1]

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

                    accumulator_nl::ComplexF64 = 0.0 + 0.0im
                    accumulator_li::ComplexF64 = 0.0 + 0.0im

                    # from H+L
                    for m1=max(Λ+1,m-Λ):1:min(M,m)
                        n2min = m1 == m ? 1 : -N
                        for n1=max(-N,n-N):1:min(n-n2min,N)

                            accumulator_nl += temp_nl[n1+ny,m1-Λ,n+ny,m-Λ]*u.x[2][n1+ny,m1-Λ,n3+ny,m3-Λ]
                            accumulator_li += temp_li[n1+ny,m1-Λ,n+ny,m-Λ]*u.x[2][n1+ny,m1-Λ,n3+ny,m3-Λ]

                        end
                    end

                    # from H-L
                    for m1=max(Λ+1,m):1:min(M,m+Λ)
                        n2max = m1 == m ? -1 : N
                        for n1=max(-N,n-n2max):1:min(n+N,N)

                            accumulator_nl += temp_nl[n1+ny,m1-Λ,n+ny,m-Λ]*u.x[2][n1+ny,m1-Λ,n3+ny,m3-Λ]
                            accumulator_li += temp_li[n1+ny,m1-Λ,n+ny,m-Λ]*u.x[2][n1+ny,m1-Λ,n3+ny,m3-Λ]

                        end
                    end

                    dΘ[n+ny,m-Λ,n3+ny,m3-Λ] = accumulator_nl + accumulator_li

                end
            end
        end
    end

    du.x[1] .= dζ

    for m=Λ+1:1:M
        for n=-N:1:N
            for m3=Λ+1:1:M
                for n3=-N:1:N

                    du.x[2][n+ny,m-Λ,n3+ny,m3-Λ] = dΘ[n+ny,m-Λ,n3+ny,m3-Λ] + conj(dΘ[n3+ny,m3-Λ,n+ny,m-Λ])

                end
            end
        end
    end

end
