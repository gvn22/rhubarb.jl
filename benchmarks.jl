using OrdinaryDiffEq
using DiffEqCallbacks: DiscreteCallback
using RecursiveArrayTools
using FFTW
using LinearAlgebra
using Plots: plot,plot!,plotlyjs,pyplot,savefig
using BenchmarkTools
# ENV["JULIA_DEBUG"] = Main

include("coefficients.jl")
include("equations.jl")
include("ics.jl")
include("tools.jl")
include("solvers.jl")
include("analysis.jl")

# tests
lx = 4.0*Float64(pi);
ly = 2.0*Float64(pi);
nx = 8;
ny = 8;

Ω = 2.0*Float64(pi)
θ = Float64(pi)/3.0
β = 2.0*Ω*sin(θ)
Ξ = 0.2*Ω
τ = 10.0/Ω

ζ0 = ic_pert_eqm(lx,ly,nx,ny,Ξ); # one ic for all
tspan = (0.0,200.0)

A = acoeffs(ly,ny,Ξ,τ)
B = bcoeffs(lx,ly,nx,ny,β,τ)
Cp,Cm = ccoeffs(lx,ly,nx,ny)

p = [nx,ny,A,B,Cp,Cm]
# @info "Unoptimized NL equations on $(nx-1)x$(ny-1) grid"
# prob = ODEProblem(nl_eqs!,ζ0,tspan,p)
# display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))
# @info "Removed dζ from NL equations on $(nx-1)x$(ny-1) grid"
# prob = ODEProblem(nl_eqs2!,ζ0,tspan,p)
# display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))
# @info "Running NL using @views"
# prob = ODEProblem(nl_eqs3!,ζ0,tspan,p)
# display(@benchmark sol1 = solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))
# @info "Running NL using @inbounds"
# prob = ODEProblem(nl_eqs4!,ζ0,tspan,p)
# display(@benchmark sol1 = solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))

@info "Latest optimised NL equations on $(nx-1)x$(ny-1) grid"
prob = ODEProblem(nl_eqs4!,ζ0,tspan,p)
display(@benchmark sol1 = solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))

Λ = 2
A = acoeffs(ly,ny,Ξ,τ)
B = bcoeffs(lx,ly,nx,ny,β,τ)
Cp,Cm = ccoeffs(lx,ly,nx,ny,Λ)
p = [nx,ny,Λ,A,B,Cp,Cm]

# @info "Unoptimized GQL equations on $(nx-1)x$(ny-1) grid"
# prob = ODEProblem(gql_eqs!,ζ0,tspan,p)
# display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))
# @info "Optimized GQL equations on $(nx-1)x$(ny-1) grid"
# prob = ODEProblem(gql_eqs2!,ζ0,tspan,p)
# display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))
# @info "Removing @views from GQL equations on $(nx-1)x$(ny-1) grid"
# prob = ODEProblem(gql_eqs3!,ζ0,tspan,p)
# display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))
# @info "Further removing bounds checking from GQL equations on $(nx-1)x$(ny-1) grid"
# prob = ODEProblem(gql_eqs4!,ζ0,tspan,p)
# display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))

@info "Latest optimized GQL equations on $(nx-1)x$(ny-1) grid"
prob = ODEProblem(gql_eqs4!,ζ0,tspan,p)
display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))

# GCE2
u0 = ic_cumulants(nx,ny,Λ,ζ0)

# @info "Unoptimized GCE2($Λ) equations on $(nx-1)x$(ny-1) grid"
# prob = ODEProblem(gce2_eqs!,u0,tspan,p)
# display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=1000,save_start=false,save_everystep=false,saveat=20))
# @info "Removed dζ from GCE2($Λ) equations on $(nx-1)x$(ny-1) grid"
# prob = ODEProblem(gce2_eqs2!,u0,tspan,p)
# display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=1000,save_start=false,save_everystep=false,saveat=20))
# @info "Removed dθ from GCE2($Λ) equations on $(nx-1)x$(ny-1) grid"
# prob = ODEProblem(gce2_eqs3!,u0,tspan,p)
# display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=1000,save_start=false,save_everystep=false,saveat=20))

# @info "Current (develop branch) GCE2($Λ) equations on $(nx-1)x$(ny-1) grid"
# prob = ODEProblem(gce2_eqs3!,u0,tspan,p);
# display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))
# @info "Removed ALL allocations and bounds checking in GCE2($Λ) equations on $(nx-1)x$(ny-1) grid"
# p = [nx,ny,Λ,A,B,Cp,Cm,fill!(similar(u0.x[2]),0)]
# prob = ODEProblem(gce2_eqs4!,u0,tspan,p)
# display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))
# @info "Further avoiding permutedims in GCE2($Λ) equations on $(nx-1)x$(ny-1) grid"
# p = [nx,ny,Λ,A,B,Cp,Cm,fill!(similar(u0.x[1]),0),fill!(similar(u0.x[2]),0),fill!(similar(u0.x[2]),0)]
# prob = ODEProblem(gce2_eqs5!,u0,tspan,p)
# display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))

@info "Latest optimized GCE2($Λ) equations on $(nx-1)x$(ny-1) grid"
p = [nx,ny,Λ,A,B,Cp,Cm,fill!(similar(u0.x[1]),0),fill!(similar(u0.x[2]),0),fill!(similar(u0.x[2]),0)]
prob = ODEProblem(gce2_eqs5!,u0,tspan,p)
display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))

# @info "Using Tsit5() for GCE2($Λ) equations on $(nx-1)x$(ny-1) grid"
# p = [nx,ny,Λ,A,B,Cp,Cm,fill!(similar(u0.x[1]),0),fill!(similar(u0.x[2]),0),fill!(similar(u0.x[2]),0)]
# prob = ODEProblem(gce2_eqs5!,u0,tspan,p)
# display(@benchmark solve(prob,Tsit5(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))
#
# @info "Using BS3 for GCE2($Λ) equations on $(nx-1)x$(ny-1) grid"
# p = [nx,ny,Λ,A,B,Cp,Cm,fill!(similar(u0.x[1]),0),fill!(similar(u0.x[2]),0),fill!(similar(u0.x[2]),0)]
# prob = ODEProblem(gce2_eqs5!,u0,tspan,p)
# display(@benchmark solve(prob,BS3(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))
#
# @info "Using RK4() for GCE2($Λ) equations on $(nx-1)x$(ny-1) grid"
# p = [nx,ny,Λ,A,B,Cp,Cm,fill!(similar(u0.x[1]),0),fill!(similar(u0.x[2]),0),fill!(similar(u0.x[2]),0)]
# prob = ODEProblem(gce2_eqs5!,u0,tspan,p)
# display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))
# @time solve(prob,RK4(),dt=0.001,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20)
#
# @info "Using @fastmath for GCE2($Λ) equations on $(nx-1)x$(ny-1) grid"
# p = [nx,ny,Λ,A,B,Cp,Cm,fill!(similar(u0.x[1]),0),fill!(similar(u0.x[2]),0),fill!(similar(u0.x[2]),0)]
# prob = ODEProblem(gce2_eqs6!,u0,tspan,p)
# display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))
