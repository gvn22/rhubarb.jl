using OrdinaryDiffEq
using RecursiveArrayTools,DiffEqCallbacks
using FFTW, LinearAlgebra
using ODEInterfaceDiffEq
using Plots
using TimerOutputs

include("coefficients.jl")
include("equations.jl")
include("ics.jl")
include("tools.jl")
include("analysis.jl")

function exec(lx::Float64,ly::Float64,nx::Int,ny::Int,T::Float64)

    # u0 = rand(ComplexF64,2*ny-1,nx)
    tspan = (0.0,T)
    u0 = ic_rand(lx,ly,nx,ny)

    A = acoeffs(ly,ny)
    B = bcoeffs(lx,ly,nx,ny)
    Cp,Cm = ccoeffs(lx,ly,nx,ny)
    p = [nx,ny,A,B,Cp,Cm]

    prob = ODEProblem(nl_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=50)

    return sol

end

function exec(lx::Float64,ly::Float64,nx::Int,ny::Int,T::Float64,Ω::Float64,θ::Float64)

    # u0 = rand(ComplexF64,2*ny-1,nx)
    tspan = (0.0,T)
    u0 = ic_rand(lx,ly,nx,ny)

    A = acoeffs(ly,ny)
    B = bcoeffs(lx,ly,nx,ny,Ω,θ)
    Cp,Cm = ccoeffs(lx,ly,nx,ny)
    p = [nx,ny,A,B,Cp,Cm]

    prob = ODEProblem(nl_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=50)

    return sol

end

function exec(lx::Float64,ly::Float64,nx::Int,ny::Int,T::Float64,νn::Float64)

    # u0 = rand(ComplexF64,2*ny-1,nx)
    tspan = (0.0,T)
    u0 = ic_rand(lx,ly,nx,ny)

    A = acoeffs(ly,ny)
    B = bcoeffs(lx,ly,nx,ny,νn)
    Cp,Cm = ccoeffs(lx,ly,nx,ny)
    p = [nx,ny,A,B,Cp,Cm]

    prob = ODEProblem(nl_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=50)

    return sol

end

function exec(lx::Float64,ly::Float64,nx::Int,ny::Int,T::Float64,Ω::Float64,θ::Float64,νn::Float64)

    # u0 = rand(ComplexF64,2*ny-1,nx)
    u0 = ic_rand(lx,ly,nx,ny)
    tspan = (0.0,T)

    A = acoeffs(ly,ny)
    B = bcoeffs(lx,ly,nx,ny,Ω,θ,νn)
    Cp,Cm = ccoeffs(lx,ly,nx,ny)
    p = [nx,ny,A,B,Cp,Cm]

    prob = ODEProblem(nl_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=50)

    return sol

end

function exec(lx::Float64,ly::Float64,nx::Int,ny::Int,T::Float64,Ω::Float64,θ::Float64,νn::Float64,Δθ::Float64,τ::Float64)

    tspan = (0.0,T)

    A = acoeffs(ly,ny,Ω,Δθ,τ)
    u0 = ic_eqm(lx,ly,nx,ny,A) + ic_rand(lx,ly,nx,ny)

    B = bcoeffs(lx,ly,nx,ny,Ω,θ,νn,τ)
    Cp,Cm = ccoeffs(lx,ly,nx,ny)
    p = [nx,ny,A,B,Cp,Cm]

    prob = ODEProblem(nl_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=50)

    return sol

end

function gql(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,T::Float64)

    # u0 = rand(ComplexF64,2*ny-1,nx)
    tspan = (0.0,T)
    u0 = ic_rand(lx,ly,nx,ny)

    A = acoeffs(ly,ny)
    B = bcoeffs(lx,ly,nx,ny)
    Cp,Cm = ccoeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,A,B,Cp,Cm]

    prob = ODEProblem(gql_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=50)

    return sol

end

function gql(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,T::Float64,νn::Float64)

    # u0 = rand(ComplexF64,2*ny-1,nx)
    tspan = (0.0,T)
    u0 = ic_rand(lx,ly,nx,ny)

    A = acoeffs(ly,ny)
    B = bcoeffs(lx,ly,nx,ny,νn)
    Cp,Cm = ccoeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,A,B,Cp,Cm]

    prob = ODEProblem(gql_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=50)

    return sol

end

function gql(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,T::Float64,Ω::Float64,θ::Float64)

    # u0 = rand(ComplexF64,2*ny-1,nx)
    tspan = (0.0,T)
    u0 = ic_rand(lx,ly,nx,ny)

    A = acoeffs(ly,ny)
    B = bcoeffs(lx,ly,nx,ny,Ω,θ)
    Cp,Cm = ccoeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,A,B,Cp,Cm]

    prob = ODEProblem(gql_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=50)

    return sol

end

function gql(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,T::Float64,Ω::Float64,θ::Float64,νn::Float64)

    # u0 = rand(ComplexF64,2*ny-1,nx)
    tspan = (0.0,T)
    u0 = ic_rand(lx,ly,nx,ny)

    A = acoeffs(ly,ny)
    B = bcoeffs(lx,ly,nx,ny,Ω,θ,νn)
    Cp,Cm = ccoeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,A,B,Cp,Cm]

    prob = ODEProblem(gql_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=50)

    return sol

end

function gql(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,T::Float64,Ω::Float64,θ::Float64,νn::Float64,Δθ::Float64,τ::Float64)

    # u0 = rand(ComplexF64,2*ny-1,nx)
    tspan = (0.0,T)
    A = acoeffs(ly,ny,Ω,Δθ,τ)
    u0 = ic_eqm(lx,ly,nx,ny,A) + ic_rand(lx,ly,nx,ny)

    B = bcoeffs(lx,ly,nx,ny,Ω,θ,νn,τ)
    Cp,Cm = ccoeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,A,B,Cp,Cm]

    prob = ODEProblem(gql_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=50)

    return sol

end

function gce2(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,T::Float64)

    # u0 = rand(ComplexF64,2*ny-1,nx)
    tspan = (0.0,T)
    u0 = ic_cumulants(nx,ny,Λ,ic_rand(lx,ly,nx,ny))

    A = acoeffs(ly,ny)
    B = bcoeffs(lx,ly,nx,ny)
    Cp,Cm = ccoeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,A,B,Cp,Cm]

    prob = ODEProblem(gce2_eqs!,u0,tspan,p)
    poschecktimes = range(1.0,T,step=10.0)
    condition(u,t,integrator) = t ∈ poschecktimes && !ispositive(u.x[2],nx,ny,Λ)
    affect!(integrator) = positivity!(integrator.u.x[2],nx,ny,Λ)
    cb = PresetTimeCallback(poschecktimes,affect!)

    @time sol = solve(prob,RK4(),callback=cb,adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=50)

    return sol

end

function gce2(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,T::Float64,Ω::Float64,θ::Float64)

    # u0 = rand(ComplexF64,2*ny-1,nx)
    tspan = (0.0,T)
    u0 = ic_cumulants(nx,ny,Λ,ic_rand(lx,ly,nx,ny))

    A = acoeffs(ly,ny)
    B = bcoeffs(lx,ly,nx,ny,Ω,θ)
    Cp,Cm = ccoeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,A,B,Cp,Cm]

    prob = ODEProblem(gce2_eqs!,u0,tspan,p)
    poschecktimes = range(1.0,T,step=10.0)
    condition(u,t,integrator) = t ∈ poschecktimes && !ispositive(u.x[2],nx,ny,Λ)
    affect!(integrator) = positivity!(integrator.u.x[2],nx,ny,Λ)
    cb = PresetTimeCallback(poschecktimes,affect!)

    @time sol = solve(prob,RK4(),callback=cb,adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=50)

    return sol

end

function gce2(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,T::Float64,νn::Float64)

    # u0 = rand(ComplexF64,2*ny-1,nx)
    tspan = (0.0,T)
    u0 = ic_cumulants(nx,ny,Λ,ic_rand(lx,ly,nx,ny))

    A = acoeffs(ly,ny)
    B = bcoeffs(lx,ly,nx,ny,νn)
    Cp,Cm = ccoeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,A,B,Cp,Cm]

    prob = ODEProblem(gce2_eqs!,u0,tspan,p)
    poschecktimes = range(1.0,T,step=10.0)
    condition(u,t,integrator) = t ∈ poschecktimes && !ispositive(u.x[2],nx,ny,Λ)
    affect!(integrator) = positivity!(integrator.u.x[2],nx,ny,Λ)
    cb = PresetTimeCallback(poschecktimes,affect!)

    @time sol = solve(prob,RK4(),callback=cb,adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=50)

    return sol

end

function gce2(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,T::Float64,Ω::Float64,θ::Float64,νn::Float64)

    # u0 = rand(ComplexF64,2*ny-1,nx)
    tspan = (0.0,T)
    u0 = ic_cumulants(nx,ny,Λ,ic_rand(lx,ly,nx,ny))

    A = acoeffs(ly,ny)
    B = bcoeffs(lx,ly,nx,ny,Ω,θ,νn)
    Cp,Cm = ccoeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,A,B,Cp,Cm]

    prob = ODEProblem(gce2_eqs!,u0,tspan,p)
    poschecktimes = range(1.0,T,step=10.0)
    condition(u,t,integrator) = t ∈ poschecktimes && !ispositive(u.x[2],nx,ny,Λ)
    affect!(integrator) = positivity!(integrator.u.x[2],nx,ny,Λ)
    cb = PresetTimeCallback(poschecktimes,affect!)

    @time sol = solve(prob,RK4(),callback=cb,adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=50)

    return sol

end

function gce2(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,T::Float64,Ω::Float64,θ::Float64,νn::Float64,Δθ::Float64,τ::Float64)

    # u0 = rand(ComplexF64,2*ny-1,nx)
    tspan = (0.0,T)

    A = acoeffs(ly,ny,Ω,Δθ,τ)
    u0 = ic_cumulants(nx,ny,Λ,ic_eqm(lx,ly,nx,ny,A) + ic_rand(lx,ly,nx,ny))

    B = bcoeffs(lx,ly,nx,ny,Ω,θ,νn,τ)
    Cp,Cm = ccoeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,A,B,Cp,Cm]

    prob = ODEProblem(gce2_eqs!,u0,tspan,p)
    # poschecktimes = range(1.0,T,step=10.0)
    poschecktimes = LinRange(1.0,T,20)
    condition(u,t,integrator) = t ∈ poschecktimes && !ispositive(u.x[2],nx,ny,Λ)
    affect!(integrator) = positivity!(integrator.u.x[2],nx,ny,Λ)
    cb = PresetTimeCallback(poschecktimes,affect!)

    @time sol = solve(prob,RK4(),callback=cb,adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=50)

    return sol

end

function exec(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,β::Float64,ν::Float64,t_end::Float64,u0::Array{ComplexF64,2})

    u0_low = u0[:,1:Λ+1]
    for n = 1:1:ny-1
        u0_low[n,1] = u0_low[2*ny - n,1]
    end
    u0_high = zeros(ComplexF64,2*ny-1,nx - Λ - 1,2*ny-1,nx - Λ - 1)
    for m1=Λ+1:1:nx-1
        for n1=-ny+1:1:ny-1
            for m2=Λ+1:1:nx-1
                for n2=-ny+1:1:ny-1

                    u0_high[n2+ny,m2-Λ,n1+ny,m1-Λ] = u0[n2+ny,m2+1]*conj(u0[n1+ny,m1+1])

                end
            end
        end
    end
    u0 = ArrayPartition(u0_low,u0_high)
    tspan = (0.0,t_end)
    ω,v,v4 = l_coeffs(lx,ly,nx,ny,β,ν)
    Cp,Cm = gql_coeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,ω,v4,Cp,Cm]
    prob = ODEProblem(gce2_eqs!,u0,tspan,p)

    # @time sol3 = solve(prob,Tsit5(),adaptive=true,reltol=1e-7,abstol=1e-7,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=10)
    @time sol = solve(prob,RK4(),alg_hints=:stiff,dt=0.001,adaptive=false,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=10)

    return sol
end

# solve NL
function exec(lx::Float64,ly::Float64,nx::Int,ny::Int,Ω::Float64,ν::Float64,τ::Float64,u0::Array{ComplexF64,2})

    Ξ = 0.6*Ω
    Δθ = 0.05
    θ = Float64(pi)/6.0
    β = 2.0*Ω*cos(θ)

    ujet = c_coeffs(lx,ly,nx,ny,Ξ,Δθ)
    ω,v,v4 = l_coeffs(lx,ly,nx,ny,β,ν)
    Cp,Cm = nl_coeffs(lx,ly,nx,ny)

    p = [nx,ny,ujet,τ,ω,v4,Cp,Cm]
    tspan = (0.0,500.0)

    prob = ODEProblem(nl_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=50)

    return sol
end

# solve GQL
function exec(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,Ω::Float64,ν::Float64,τ::Float64,u0::Array{ComplexF64,2})

    Ξ = 0.6*Ω
    Δθ = 0.05
    θ = Float64(pi)/6.0
    β = 2.0*Ω*cos(θ)

    ujet = c_coeffs(lx,ly,nx,ny,Ξ,Δθ)
    ω,v,v4 = l_coeffs(lx,ly,nx,ny,β,ν)
    Cp,Cm = gql_coeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,ujet,τ,ω,v4,Cp,Cm]
    tspan = (0.0,500.0)

    prob = ODEProblem(gql_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=50)

    return sol
end

# solve GCE2
function exec_gce2(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,Ω::Float64,ν::Float64,τ::Float64,u0::Array{ComplexF64,2})

    Ξ = 0.6*Ω
    Δθ = 0.05
    θ = Float64(pi)/6.0
    β = 2.0*Ω*cos(θ)

    u0_low = u0[:,1:Λ+1]
    for n = 1:1:ny-1
        u0_low[n,1] = u0_low[2*ny - n,1]
    end
    u0_high = zeros(ComplexF64,2*ny-1,nx - Λ - 1,2*ny-1,nx - Λ - 1)
    for m1=Λ+1:1:nx-1
        for n1=-ny+1:1:ny-1
            for m2=Λ+1:1:nx-1
                for n2=-ny+1:1:ny-1

                    u0_high[n2+ny,m2-Λ,n1+ny,m1-Λ] = u0[n2+ny,m2+1]*conj(u0[n1+ny,m1+1])

                end
            end
        end
    end
    u0 = ArrayPartition(u0_low,u0_high)

    ujet = c_coeffs(lx,ly,nx,ny,Ξ,Δθ)
    ω,v,v4 = l_coeffs(lx,ly,nx,ny,β,ν)
    Cp,Cm = gql_coeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,ujet,τ,ω,v4,Cp,Cm]
    tspan = (0,500.0)
    prob = ODEProblem(gce2_eqs!,u0,tspan,p)

    poschecktimes = LinRange(1.0,tspan[2],50)
    condition(u,t,integrator) = t ∈ poschecktimes && !ispositive(u.x[2],nx,ny,Λ)
    affect!(integrator) = positivity!(integrator.u.x[2],nx,ny,Λ)
    cb = PresetTimeCallback(poschecktimes,affect!)

    @time sol = solve(prob,RK4(),callback=cb,adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=50)
    # integrator = init(prob,Tsit5())

    return sol
end

# global code
lx = 4.0*Float64(pi)
ly = 2.0*Float64(pi)
nx = 4
ny = 4
T = 100.0
Ω = 2.0*Float64(pi)
θ = Float64(pi)/6.0
νn = 1.0
Δθ = 0.05
τ = 2.0

plotly()

xx = LinRange(-lx/2,lx/2,2*nx-1)
yy = LinRange(-ly/2,ly/2,2*ny-1)
angles = yy*180.0/ly
modes = ["0" "1" "2" "3"]

sol1 = exec(lx,ly,nx,ny,T,Ω,θ,νn,Δθ,τ)

E,Z = energy(lx,ly,nx,ny,sol1.u)
Plots.plot(sol1.t,E,linewidth=2,legend=:bottom,xaxis="t",label="E")
Plots.plot!(sol1.t,Z,linewidth=2,legend=:bottom,xaxis="t",label="Z")

uxy = inversefourier(nx,ny,sol1.u)
Plots.plot(xx,yy,uxy[:,:,begin],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
Plots.plot(xx,yy,uxy[:,:,end],st=:contourf,color=:bwr,xaxis="x",yaxis="y")

P,O = zonalpower(lx,ly,nx,ny,sol1.u)
Plots.plot(sol1.t,P,yscale=:log,labels=modes,legend=:outertopright,linewidth=2)
Plots.plot(sol1.t,O,yscale=:log,labels=modes,legend=:outertopright,linewidth=2)

A = meanvorticity(lx,ly,nx,ny,sol1.u)
Plots.plot(angles,A[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="NL")
Plots.plot(sol1.t,angles,A',yaxis="θ",st=:contourf,color=:bwr,xaxis="t")

Λ = 1

sol2 = gql(lx,ly,nx,ny,Λ,T,Ω,θ,νn,Δθ,τ)
E,Z = energy(lx,ly,nx,ny,sol2.u)
Plots.plot(sol2.t,E,linewidth=2,legend=:bottom,label="E")
Plots.plot!(sol2.t,Z,linewidth=2,legend=:bottom,label="Z")

uxy = inversefourier(nx,ny,sol2.u)
Plots.plot(xx,yy,uxy[:,:,begin],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
Plots.plot(xx,yy,uxy[:,:,end],st=:contourf,color=:bwr,xaxis="x",yaxis="y")

P,O = zonalpower(lx,ly,nx,ny,sol2.u)
Plots.plot(sol2.t,P,yscale=:log,labels=modes,legend=:outertopright,linewidth=2)
Plots.plot(sol2.t,O,yscale=:log,labels=modes,legend=:outertopright,linewidth=2)

A = meanvorticity(lx,ly,nx,ny,sol2.u)
Plots.plot(angles,A[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="NL")
Plots.plot(sol2.t,angles,A',yaxis="θ",st=:contourf,color=:bwr,xaxis="t")

sol3 = gce2(lx,ly,nx,ny,Λ,T,Ω,θ,νn,Δθ,τ)
E,Z = energy(lx,ly,nx,ny,Λ,sol3.u)
Plots.plot(sol3.t,E,linewidth=2,legend=:bottom,label="E")
Plots.plot!(sol3.t,Z,linewidth=2,legend=:bottom,label="Z")

P,O = zonalpower(lx,ly,nx,ny,Λ,sol3.u)
Plots.plot(sol3.t,P,yscale=:log,labels=modes,legend=:outertopright,linewidth=2)
Plots.plot(sol3.t,O,yscale=:log,labels=modes,legend=:outertopright,linewidth=2)

A = meanvorticity(lx,ly,nx,ny,Λ,sol3.u)
Plots.plot(angles,A[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="GCE2")
Plots.plot(sol3.t,angles,A',yaxis="θ",st=:contourf,color=:bwr,xaxis="t")

uxy = inversefourier(nx,ny,Λ,sol3.u)
Plots.plot(xx,yy,uxy[:,:,begin],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
Plots.plot(xx,yy,uxy[:,:,end],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
