using OrdinaryDiffEq,DiffEqCallbacks
using RecursiveArrayTools
using FFTW,LinearAlgebra
using Plots,Logging
using TimerOutputs
# ENV["JULIA_DEBUG"] = Main

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
# θ = 0.0
θ = Float64(pi)/6.0
Δθ = 0.05
# νn = 0.0
# τ = 2.0
u0 = ic_eqm(lx,ly,nx,ny,Ω,Δθ) + ic_rand(lx,ly,nx,ny)/10.0

xx = LinRange(-lx/2,lx/2,2*nx-1)
yy = LinRange(-ly/2,ly/2,2*ny-1)
angles = yy*180.0/ly
zones = reshape(["$i" for i = 0:1:nx-1],1,nx)
modes = reshape(["($j,$i)" for j=0:1:nx-1 for i=-(ny-1):1:ny-1],1,nx*(2*ny-1))

# plotlyjs();
pyplot();
dn = "tests/2x2/icjet+randby10+beta+nl+tropic/"

## NL

# sol1 = exec(lx,ly,nx,ny,T,u0)
sol1 = exec(lx,ly,nx,ny,T,Ω,θ,u0)

E1,Z1 = energy(lx,ly,nx,ny,sol1.u)
_ez = Plots.plot(sol1.t,E1,linewidth=2,label="E")
_ez = Plots.plot!(sol1.t,Z1,linewidth=2,legend=:right,yaxis="Energy,Enstrophy",xaxis="Time",label="Z")
Plots.savefig(_ez,dn*"NL_ez_t.png")

P1,O1 = zonalpower(lx,ly,nx,ny,sol1.u)
_p = Plots.plot(sol1.t,P1,yscale=:log10,xaxis=("Time"),yaxis=("Energy in Mode"),labels=zones,legend=:right,linewidth=2)
Plots.savefig(_p,dn*"NL_em_t.png")

M1 = modalenergy(lx,ly,nx,ny,sol1.u);
_m = Plots.plot(sol1.t,M1,labels=modes,legend=:outerright,linewidth=2,xaxis=("Time"),yaxis="Mode Strength")
Plots.savefig(_m,dn*"NL_m_t.png")

U1 = inversefourier(nx,ny,sol1.u)
_u = Plots.plot(xx,yy,U1[:,:,begin],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
Plots.savefig(_u,dn*"NL_z_init.png")
_u = Plots.plot(xx,yy,U1[:,:,end],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
Plots.savefig(_u,dn*"NL_z_end.png")

## GQL
Λ = 1
# sol2 = gql(lx,ly,nx,ny,Λ,T,u0)
sol2 = gql(lx,ly,nx,ny,Λ,T,Ω,θ,u0)

E2,Z2 = energy(lx,ly,nx,ny,sol2.u)
_ez = Plots.plot(sol2.t,E2,linewidth=2,label="E")
_ez = Plots.plot!(sol2.t,Z2,linewidth=2,legend=:right,yaxis="Energy,Enstrophy",xaxis="Time",label="Z")
Plots.savefig(_ez,dn*"GQL_"*"$Λ"*"_ez_t.png")

P2,O2 = zonalpower(lx,ly,nx,ny,sol2.u)
_p = Plots.plot(sol2.t,P2,yscale=:log10,xaxis=("Time"),yaxis=("Energy in Mode"),labels=zones,legend=:right,linewidth=2)
Plots.savefig(_p,dn*"GQL_"*"$Λ"*"_em_t.png")

M2 = modalenergy(lx,ly,nx,ny,sol2.u)
_m = Plots.plot(sol2.t,M2,labels=modes,legend=:outerright,linewidth=2,xaxis=("Time"),yaxis="Mode Strength")
Plots.savefig(_m,dn*"GQL_"*"$Λ"*"_m_t.png")

U2 = inversefourier(nx,ny,sol2.u)
_u = Plots.plot(xx,yy,U2[:,:,begin],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
Plots.savefig(_u,dn*"GQL_"*"$Λ"*"_z_init.png")
_u = Plots.plot(xx,yy,U2[:,:,end],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
Plots.savefig(_u,dn*"GQL_"*"$Λ"*"_z_end.png")

## GCE2

# sol3 = gce2(lx,ly,nx,ny,Λ,T,u0)
sol3 = gce2(lx,ly,nx,ny,Λ,T,Ω,θ,u0)

E3,Z3 = energy(lx,ly,nx,ny,Λ,sol3.u)
_ez = Plots.plot(sol3.t,E3,linewidth=2,label="E")
_ez = Plots.plot!(sol3.t,Z3,linewidth=2,legend=:right,yaxis="Energy,Enstrophy",xaxis="Time",label="Z")
Plots.savefig(_ez,dn*"GCE2_"*"$Λ"*"_fix_ez_t.png")

P3,O3 = zonalpower(lx,ly,nx,ny,Λ,sol3.u)
_p = Plots.plot(sol3.t,P3,yscale=:log10,xaxis=("Time"),yaxis=("Energy in Mode"),labels=zones,legend=:right,linewidth=2)
Plots.savefig(_p,dn*"GCE2_"*"$Λ"*"_fix_em_t.png")

M3 = modalenergy(lx,ly,nx,ny,Λ,sol3.u)
_m = Plots.plot(sol3.t,M3,labels=modes,legend=:outerright,linewidth=2,xaxis=("Time"),yaxis="Mode Strength")
Plots.savefig(_m,dn*"GCE2_"*"$Λ"*"_fix_m_t.png")

U3 = inversefourier(nx,ny,Λ,sol3.u)
_u = Plots.plot(xx,yy,U3[:,:,begin],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
Plots.savefig(_u,dn*"GCE2_"*"$Λ"*"_fix_z_init.png")
_u = Plots.plot(xx,yy,U3[:,:,end],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
Plots.savefig(_u,dn*"GCE2_"*"$Λ"*"_fix_z_end.png")

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
# Λ = 0
# sol3_6p = gce2(lx,ly,nx,ny,Λ,T,u0)
#
# sol3_6p_dt001 = gce2(lx,ly,nx,ny,Λ,T,u0)
# P,O = zonalpower(lx,ly,nx,ny,Λ,sol3_6p_dt001.u)
# Plots.plot(sol3_6p_dt001.t,P,yscale=:log10,yaxis=("Energy in Mode m"),labels=modes,legend=:outertopright,linewidth=2)
#
# sol3_6p_dt0005 = gce2(lx,ly,nx,ny,Λ,T,u0)
# P,O = zonalpower(lx,ly,nx,ny,Λ,sol3_6p_dt0005.u)
# Plots.plot(sol3_6p_dt0005.t,P,yscale=:log10,yaxis=("Energy in Mode m"),labels=modes,legend=:outertopright,linewidth=2)
#
# sol3_7p = gce2(lx,ly,nx,ny,Λ,T,u0)
# sol3_8p = gce2(lx,ly,nx,ny,Λ,T,u0)
