## NL
function nl(lx::Float64,ly::Float64,nx::Int,ny::Int,Ξ::Float64,β::Float64,τ::Float64=0.0,
    νn::Float64=0.0;jw::Float64=0.05,ic::Array{ComplexF64,2},dt::Float64=0.01,t_end::Float64=1000.0,ts_alg=RK4,savefreq::Int=20,kwargs...)
    A = acoeffs(ly,ny,Ξ,τ,jw=jw)
    B = bcoeffs(lx,ly,nx,ny,β,τ,νn)
    Cp,Cm = ccoeffs(lx,ly,nx,ny)
    p = [nx,ny,A,B,Cp,Cm]
    tspan = (0.0,t_end)
    prob = ODEProblem(nl_eqs4!,ic,tspan,p)
    @info "Solving NL equations on $(nx-1)x$(ny-1) grid"
    solve(prob,ts_alg(),dt=dt,adaptive=false,progress=true,progress_steps=10000,save_start=true,saveat=savefreq,save_everystep=savefreq==1 ? true : false)
end

## GQL
function gql(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,Ξ::Float64,β::Float64,τ::Float64=0.0,
    νn::Float64=0.0;jw::Float64=0.05,ic::Array{ComplexF64,2},dt::Float64=0.01,t_end::Float64=1000.0,ts_alg=RK4,savefreq::Int=20,kwargs...)
    A = acoeffs(ly,ny,Ξ,τ,jw=jw)
    B = bcoeffs(lx,ly,nx,ny,β,τ,νn)
    Cp,Cm = ccoeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,A,B,Cp,Cm]
    tspan = (0.0,t_end)
    if ts_alg == ETDRK4 || ts_alg == ETDRK2
        LO = DiffEqOperators.DiffEqArrayOperator(Diagonal(reshape(B,(2*ny-1)*nx)))
        prob = SplitODEProblem(LO,gql_eqs4_f2,reshape(ic,(2*ny-1)*nx),tspan,p)
        @info "Solving GQL equations on $(nx-1)x$(ny-1) grid with Λ = $Λ using ETDRK4 method"
        ts_alg = ETDRK4(autodiff=false)
    else
        prob = ODEProblem(gql_eqs4!,ic,tspan,p)
        @info "Solving GQL equations on $(nx-1)x$(ny-1) grid with Λ = $Λ using $ts_alg method"
        ts_alg = ts_alg()
    end
    solve(prob,ts_alg,dt=dt,adaptive=false,progress=true,progress_steps=10000,save_start=true,save_everystep=false,dense=false,saveat=savefreq)
end

## GCE2
function gce2(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,Ξ::Float64,β::Float64,τ::Float64=0.0,
    νn::Float64=0.0;jw::Float64=0.05,ic::Array{ComplexF64,2},dt::Float64=0.01,t_end::Float64=1000.0,ts_alg=RK4,poscheck::Bool=false,savefreq::Int=20,poscheckfreq::Float64=50.0,kwargs...)
    A = acoeffs(ly,ny,Ξ,τ,jw=jw)
    B = bcoeffs(lx,ly,nx,ny,β,τ,νn)
    Cp,Cm = ccoeffs(lx,ly,nx,ny,Λ)
    tspan = (0.0,t_end)
    u0 = ic_cumulants(nx,ny,Λ,ic)
    p = [nx,ny,Λ,A,B,Cp,Cm,fill!(similar(u0.x[1]),0),fill!(similar(u0.x[2]),0),fill!(similar(u0.x[2]),0)]
    # solver specific problem setup
    if ts_alg == ETDRK4 || ts_alg == ETDRK2
        d1 = fill!(similar(u0.x[1]),0)
        @inbounds for m = 0:Λ
            nmin = m == 0 ? 1 : -(ny-1)
            @inbounds for n=nmin:ny-1
                d1[n+ny,m+1] = B[n+ny,m+1]
            end
        end
        d2 = fill!(similar(u0.x[2]),0)
        @inbounds for m3=Λ+1:nx-1
            @inbounds for n3=-(ny-1):ny-1
                @inbounds for m=Λ+1:nx-1
                    @inbounds for n=-(ny-1):ny-1
                        d2[n+ny,m-Λ,n3+ny,m3-Λ] = B[n+ny,m+1] + conj(B[n3+ny,m3+1])
                    end
                end
            end
        end
        LO = DiffEqOperators.DiffEqArrayOperator(Diagonal(ArrayPartition(d1,d2)))
        prob = SplitODEProblem(LO,gce2_eqs5_f2,u0,tspan,p)
        @info "Solving GCE2 equations on $(nx-1)x$(ny-1) grid with Λ = $Λ using $ts_alg method"
        ts_alg = ETDRK4(autodiff=false)
    else
        prob = ODEProblem(gce2_eqs5!,u0,tspan,p)
        @info "Solving GCE2 equations on $(nx-1)x$(ny-1) grid with Λ = $Λ using $ts_alg method"
        ts_alg = ts_alg()
    end
    # callback based solver setup
    if poscheck && Λ < nx - 1
        poschecktimes = [tt for tt in range(1.0,t_end,step=poscheckfreq)]
        condition(u,t,integrator) = t ∈ poschecktimes && !ispositive(u.x[2],nx,ny,Λ)
        affect!(integrator) = positivity!(integrator.u.x[2],nx,ny,Λ)
        cb = DiscreteCallback(condition,affect!,save_positions=(false,false))
        solve(prob,ts_alg,callback=cb,tstops=poschecktimes,dt=dt,adaptive=false,progress=true,progress_steps=10000,save_start=true,save_everystep=false,dense=false,saveat=savefreq)
    else
        solve(prob,ts_alg,dt=dt,adaptive=false,progress=true,progress_steps=10000,save_start=true,save_everystep=false,saveat=savefreq)
    end
end
