using DifferentialEquations
using RecursiveArrayTools
using FFTW, LinearAlgebra
using Plots
using TimerOutputs
using Logging
# ENV["JULIA_DEBUG"] = Main

include("coefficients.jl")

include("equations.jl")
include("ics.jl")
include("tools.jl")
include("solvers.jl")

include("analysis.jl")

## Parameters

lx = 4.0*Float64(pi)
ly = 2.0*Float64(pi)
nx = 6
ny = 6
T = 500.0
# u0 = ic_rand(lx,ly,nx,ny)

Ω = 2.0*Float64(pi)
θ = Float64(pi)/6.0
νn = 0.0
Δθ = 0.05
τ = 2.0
u0 = ic_eqm(lx,ly,nx,ny,Ω,Δθ) + ic_rand(lx,ly,nx,ny)/100.0

plotlyjs()

xx = LinRange(-lx/2,lx/2,2*nx-1)
yy = LinRange(-ly/2,ly/2,2*ny-1)
angles = yy*180.0/ly
modes = ["0" "1" "2" "3" "4" "5" "6"]

## NL

sol1 = exec(lx,ly,nx,ny,T,Ω,θ,νn,Δθ,τ,u0)

E,Z = energy(lx,ly,nx,ny,sol1.u)
Plots.plot(sol1.t,E,linewidth=2,legend=:bottom,xaxis="t",label="E")
Plots.plot!(sol1.t,Z,linewidth=2,legend=:bottom,xaxis="t",label="Z")

P,O = zonalpower(lx,ly,nx,ny,sol1.u)
Plots.plot(sol1.t,P,yscale=:log10,xaxis=("Time"),yaxis=("Energy in Mode m"),labels=modes,legend=:outertopright,linewidth=2)
# Plots.plot(sol1.t,O,yscale=:log,labels=modes,legend=:outertopright,linewidth=2)

uxy = inversefourier(nx,ny,sol1.u)
Plots.plot(xx,yy,uxy[:,:,begin],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
Plots.plot(xx,yy,uxy[:,:,end],st=:contourf,color=:bwr,xaxis="x",yaxis="y")

A1 = meanvorticity(lx,ly,nx,ny,sol1.u)
Plots.plot(angles,A1[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="NL")
Plots.plot(sol1.t,angles,A1',yaxis="θ",st=:contourf,color=:bwr,xaxis="t")

## GQL

Λ = 0

sol2 = gql(lx,ly,nx,ny,Λ,T,Ω,θ,νn,Δθ,τ,u0)
E,Z = energy(lx,ly,nx,ny,sol2.u)
Plots.plot(sol2.t,E,linewidth=2,legend=:bottom,label="E")
Plots.plot!(sol2.t,Z,linewidth=2,legend=:bottom,label="Z")

P,O = zonalpower(lx,ly,nx,ny,sol2.u)
Plots.plot(sol2.t,P,yscale=:log10,yaxis=("Energy in Mode m",(1e-3,1e3)),labels=modes,legend=:outertopright,linewidth=2)
# Plots.plot(sol2.t,O,yscale=:log,labels=modes,legend=:outertopright,linewidth=2)

uxy = inversefourier(nx,ny,sol2.u)
Plots.plot(xx,yy,uxy[:,:,begin],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
Plots.plot(xx,yy,uxy[:,:,end],st=:contourf,color=:bwr,xaxis="x",yaxis="y")

A2_0 = meanvorticity(lx,ly,nx,ny,sol2.u)
Plots.plot(angles,A[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="NL")
Plots.plot(sol2.t,angles,A2_0',yaxis="θ",st=:contourf,color=:bwr,xaxis="t")

## GCE2

sol3 = gce2(lx,ly,nx,ny,Λ,T,Ω,θ,νn,Δθ,τ,u0)
E,Z = energy(lx,ly,nx,ny,Λ,sol3.u)
Plots.plot(sol3.t,E,linewidth=2,legend=:bottom,label="E")
Plots.plot!(sol3.t,Z,linewidth=2,legend=:bottom,label="Z")

P,O = zonalpower(lx,ly,nx,ny,Λ,sol3.u)
Plots.plot(sol3.t,P,yscale=:log10,yaxis=("Energy in Mode m"),labels=modes,legend=:outertopright,linewidth=2)
# Plots.plot(sol3.t,O,yscale=:log10,labels=modes,legend=:outertopright,linewidth=2)

A3_0 = meanvorticity(lx,ly,nx,ny,Λ,sol3.u)
Plots.plot(angles,A[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="GCE2")
Plots.plot(sol3.t,angles,A3_0',yaxis="θ",st=:contourf,color=:bwr,xaxis="t")

uxy = inversefourier(nx,ny,Λ,sol3.u)
Plots.plot(xx,yy,uxy[:,:,begin],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
Plots.plot(xx,yy,uxy[:,:,end],st=:contourf,color=:bwr,xaxis="x",yaxis="y")

## vorticity plots

Plots.plot(angles,A1[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="NL")
Plots.plot!(angles,A2[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="GQL(2)")
Plots.plot!(angles,A3[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="GCE2(2)")

Plots.plot(angles,A1[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="NL")
Plots.plot!(angles,A2_1[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="GQL(1)")
Plots.plot!(angles,A3_1[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="GCE2(1)")

Plots.plot(angles,A1[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="NL")
Plots.plot!(angles,A2_0[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="GQL(0)")
Plots.plot!(angles,A3_0[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="GCE2(0)")

## tests
Λ = 0
sol3_6p = gce2(lx,ly,nx,ny,Λ,T,u0)

sol3_6p_dt001 = gce2(lx,ly,nx,ny,Λ,T,u0)
P,O = zonalpower(lx,ly,nx,ny,Λ,sol3_6p_dt001.u)
Plots.plot(sol3_6p_dt001.t,P,yscale=:log10,yaxis=("Energy in Mode m"),labels=modes,legend=:outertopright,linewidth=2)

sol3_6p_dt0005 = gce2(lx,ly,nx,ny,Λ,T,u0)
P,O = zonalpower(lx,ly,nx,ny,Λ,sol3_6p_dt0005.u)
Plots.plot(sol3_6p_dt0005.t,P,yscale=:log10,yaxis=("Energy in Mode m"),labels=modes,legend=:outertopright,linewidth=2)

sol3_7p = gce2(lx,ly,nx,ny,Λ,T,u0)
sol3_8p = gce2(lx,ly,nx,ny,Λ,T,u0)
