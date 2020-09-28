using DifferentialEquations
using RecursiveArrayTools
using FFTW, LinearAlgebra
using Plots
using TimerOutputs

include("coefficients.jl")
include("equations.jl")
include("ics.jl")
include("tools.jl")
include("solvers.jl")

include("analysis.jl")

## Parameters

lx = 2.0*Float64(pi)
ly = 2.0*Float64(pi)
nx = 2
ny = 2
T = 500.0
Ω = 2.0*Float64(pi)
θ = Float64(pi)/6.0
νn = 0.0
Δθ = 0.05
τ = 2.0

plotly()

xx = LinRange(-lx/2,lx/2,2*nx-1)
yy = LinRange(-ly/2,ly/2,2*ny-1)
angles = yy*180.0/ly
modes = ["0" "1" "2" "3" "4" "5"]

## NL

sol1 = exec(lx,ly,nx,ny,T)

E,Z = energy(lx,ly,nx,ny,sol1.u)
Plots.plot(sol1.t,E,linewidth=2,legend=:bottom,xaxis="t",label="E")
Plots.plot!(sol1.t,Z,linewidth=2,legend=:bottom,xaxis="t",label="Z")

P,O = zonalpower(lx,ly,nx,ny,sol1.u)
Plots.plot(sol1.t,P,yscale=:log10,yaxis=("Energy in Mode m"),labels=modes,legend=:outertopright,linewidth=2)
Plots.plot(sol1.t,O,yscale=:log,labels=modes,legend=:outertopright,linewidth=2)

uxy = inversefourier(nx,ny,sol1.u)
Plots.plot(xx,yy,uxy[:,:,begin],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
Plots.plot(xx,yy,uxy[:,:,end],st=:contourf,color=:bwr,xaxis="x",yaxis="y")

A = meanvorticity(lx,ly,nx,ny,sol1.u)
Plots.plot(angles,A[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="NL")
Plots.plot(sol1.t,angles,A',yaxis="θ",st=:contourf,color=:bwr,xaxis="t")

## GQL

Λ = 1

sol2 = gql(lx,ly,nx,ny,Λ,T)
E,Z = energy(lx,ly,nx,ny,sol2.u)
Plots.plot(sol2.t,E,linewidth=2,legend=:bottom,label="E")
Plots.plot!(sol2.t,Z,linewidth=2,legend=:bottom,label="Z")

P,O = zonalpower(lx,ly,nx,ny,sol2.u)
Plots.plot(sol2.t,P,yscale=:log10,yaxis=("Energy in Mode m"),labels=modes,legend=:outertopright,linewidth=2)
Plots.plot(sol2.t,O,yscale=:log,labels=modes,legend=:outertopright,linewidth=2)

uxy = inversefourier(nx,ny,sol2.u)
Plots.plot(xx,yy,uxy[:,:,begin],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
Plots.plot(xx,yy,uxy[:,:,end],st=:contourf,color=:bwr,xaxis="x",yaxis="y")

A = meanvorticity(lx,ly,nx,ny,sol2.u)
Plots.plot(angles,A[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="NL")
Plots.plot(sol2.t,angles,A',yaxis="θ",st=:contourf,color=:bwr,xaxis="t")

## GCE2
sol3 = gce2(lx,ly,nx,ny,Λ,T)
E,Z = energy(lx,ly,nx,ny,Λ,sol3.u)
Plots.plot(sol3.t,E,linewidth=2,legend=:bottom,label="E")
Plots.plot!(sol3.t,Z,linewidth=2,legend=:bottom,label="Z")

P,O = zonalpower(lx,ly,nx,ny,Λ,sol3.u)
Plots.plot(sol3.t,P,yscale=:log10,yaxis=("Energy in Mode m"),labels=modes,legend=:outertopright,linewidth=2)
Plots.plot(sol3.t,O,yscale=:log10,labels=modes,legend=:outertopright,linewidth=2)

A = meanvorticity(lx,ly,nx,ny,Λ,sol3.u)
Plots.plot(angles,A[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="GCE2")
Plots.plot(sol3.t,angles,A',yaxis="θ",st=:contourf,color=:bwr,xaxis="t")

uxy = inversefourier(nx,ny,Λ,sol3.u)
Plots.plot(xx,yy,uxy[:,:,begin],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
Plots.plot(xx,yy,uxy[:,:,end],st=:contourf,color=:bwr,xaxis="x",yaxis="y")

## tests
