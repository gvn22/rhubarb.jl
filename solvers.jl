## NL
function nl(lx::Float64,ly::Float64,nx::Int,ny::Int,Ξ::Float64,β::Float64,τ::Float64=0.0,νn::Float64=0.0;ic::Array{ComplexF64,2},dt::Float64=0.001,t_end::Float64=1000.0,kwargs...)
    A = acoeffs(ly,ny,Ξ,τ)
    B = bcoeffs(lx,ly,nx,ny,β,τ,νn)
    Cp,Cm = ccoeffs(lx,ly,nx,ny)
    p = [nx,ny,A,B,Cp,Cm]
    tspan = (0.0,t_end)
    prob = ODEProblem(nl_eqs!,ic,tspan,p)
    @info "Solving NL equations..."
    solve(prob,RK4(),dt=dt,adaptive=false,progress=true,progress_steps=10000,save_start=true,save_everystep=false,saveat=20)
end

## GQL
function gql(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,Ξ::Float64,β::Float64,τ::Float64=0.0,νn::Float64=0.0;ic::Array{ComplexF64,2},dt::Float64=0.001,t_end::Float64=1000.0)
    A = acoeffs(ly,ny,Ξ,τ)
    B = bcoeffs(lx,ly,nx,ny,β,τ,νn)
    Cp,Cm = ccoeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,A,B,Cp,Cm]
    tspan = (0.0,t_end)
    prob = ODEProblem(gql_eqs!,ic,tspan,p)
    @info "Solving GQL equations with Λ = ", Λ
    solve(prob,RK4(),dt=dt,adaptive=false,progress=true,progress_steps=10000,save_start=true,save_everystep=false,saveat=20)
end

## GCE2
function gce2(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,Ξ::Float64,β::Float64,τ::Float64=0.0,νn::Float64=0.0;ic::Array{ComplexF64,2},dt::Float64=0.001,t_end::Float64=1000.0,poscheck::Bool=false,poscheckfreq::Float64=50.0)
    A = acoeffs(ly,ny,Ξ,τ)
    B = bcoeffs(lx,ly,nx,ny,β,τ,νn)
    Cp,Cm = ccoeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,A,B,Cp,Cm]
    @info "Solving GCE2 equations with Λ = ", Λ
    tspan = (0.0,t_end)
    u0 = ic_cumulants(nx,ny,Λ,ic)
    prob = ODEProblem(gce2_eqs!,u0,tspan,p)
    if poscheck && Λ < nx - 1
        poschecktimes = [tt for tt in range(1.0,t_end,step=poscheckfreq)]
        condition(u,t,integrator) = t ∈ poschecktimes && !ispositive(u.x[2],nx,ny,Λ)
        affect!(integrator) = positivity!(integrator.u.x[2],nx,ny,Λ)
        cb = DiscreteCallback(condition,affect!,save_positions=(false,false))
        solve(prob,RK4(),callback=cb,tstops=poschecktimes,dt=dt,adaptive=false,progress=true,progress_steps=2000,save_start=true,save_everystep=false,dense=false,saveat=40)
    else
        solve(prob,RK4(),dt=dt,adaptive=false,progress=true,progress_steps=10000,save_start=true,save_everystep=false,saveat=20)
    end
end
