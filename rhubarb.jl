using OrdinaryDiffEq
using DiffEqCallbacks
using RecursiveArrayTools
using FFTW
using LinearAlgebra
using Plots: plot,plot!,plotlyjs,savefig
# ENV["JULIA_DEBUG"] = Main

include("coefficients.jl")
include("equations.jl")
include("ics.jl")
include("tools.jl")
include("solvers.jl")
include("analysis.jl")

""" Set parameters and solve
"""
lx = 4.0*Float64(pi);
ly = 2.0*Float64(pi);
nx = 4;
ny = 4;

Ω = 2.0*Float64(pi)
θ = 0.0
β = 2.0*Ω*cos(θ)
Ξ = 0.6*Ω
τ = 10.0/Ω
Λ = 1

ζ0 = ic_pert_eqm(lx,ly,nx,ny,Ξ); # one ic for all

sol1 = nl(lx,ly,nx,ny,Ξ,β,τ,ic=ζ0,dt=0.001,t_end=500.0);
sol2 = gql(lx,ly,nx,ny,Λ,Ξ,β,τ,ic=ζ0,t_end=500.0);
sol3 = gce2(lx,ly,nx,ny,Λ,Ξ,β,τ,ic=ζ0,dt=0.01,t_end=500.0,poscheck=false);

""" Create plots
"""
# plotlyjs();
pyplot();
dn = "tests/6x12/l90_j02_t20/"
mkpath(dn)
# pyplot();

## Zonal energy
zones = reshape(["$i" for i = 0:1:nx-1],1,nx);

P1,O1 = zonalenergy(lx,ly,nx,ny,sol1.u);
_p = plot(sol1.t,P1,xaxis=("Time"),yscale=:log10,yaxis=("Energy in Mode"),labels=zones,legend=:right,linewidth=2)
savefig(_p,dn*"NL_em_t.png");

P2,O2 = zonalenergy(lx,ly,nx,ny,sol2.u);
_p = plot(sol2.t,P2,xaxis=("Time"),yscale=:log10,yaxis=("Energy in Mode",(1e-2,1e3)),labels=zones,legend=:right,linewidth=2)
savefig(_p,dn*"GQL_"*"$Λ"*"_em_t.png")

P3,O3 = zonalenergy(lx,ly,nx,ny,Λ,sol3.u);
_p = plot(sol3.t,P3,xaxis=("Time"),yscale=:log10,yaxis=("Energy in Mode",(1e-2,1e3)),labels=zones,legend=:right,linewidth=2)
savefig(_p,dn*"GCE2_"*"$Λ"*"_dt01.png");

## Modal strength
modes = reshape(["($j,$i)" for j=0:1:nx-1 for i=-(ny-1):1:ny-1],1,nx*(2*ny-1));

M1 = modalstrength(lx,ly,nx,ny,sol1.u);
_m = plot(sol1.t,M1,labels=modes,legend=:outerright,linewidth=2,xaxis=("Time"),yaxis="Mode Strength")
savefig(_m,dn*"NL_m_t.png");

M2 = modalstrength(lx,ly,nx,ny,sol2.u);
_m = plot(sol2.t,M2,labels=modes,legend=:outerright,linewidth=2,xaxis=("Time"),yaxis="Mode Strength")
savefig(_m,dn*"GQL_"*"$Λ"*"_m_t.png");

M3 = modalstrength(lx,ly,nx,ny,Λ,sol3.u);
_m = plot(sol3.t,M3,labels=modes,linewidth=2,xaxis=("Time"),yaxis="Mode Strength")
savefig(_m,dn*"GCE2_"*"$Λ"*"_m_t.png")

## Spatial vorticity
xx = LinRange(-lx/2,lx/2,2*nx-1);
yy = LinRange(-ly/2,ly/2,2*ny-1);

U1 = inversefourier(nx,ny,sol1.u);
_u = plot(xx,yy,U1[:,:,begin],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
savefig(_u,dn*"NL_z_init.png");
_u = plot(xx,yy,U1[:,:,end],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
savefig(_u,dn*"NL_z_end.png");

U2 = inversefourier(nx,ny,sol2.u);
_u = plot(xx,yy,U2[:,:,begin],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
savefig(_u,dn*"GQL_"*"$Λ"*"_z_init.png")
_u = plot(xx,yy,U2[:,:,end],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
savefig(_u,dn*"GQL_"*"$Λ"*"_z_end.png")

U3 = inversefourier(nx,ny,Λ,sol3.u)
_u = plot(xx,yy,U3[:,:,begin],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
savefig(_u,dn*"GCE2_"*"$Λ"*"_z_init.png")
_u = plot(xx,yy,U3[:,:,end],st=:contourf,color=:bwr,xaxis="x",yaxis="y")
savefig(_u,dn*"GCE2_"*"$Λ"*"_z_end.png")

## Hövmoller: mean vorticity
angles = (180.0/ly)*LinRange(-ly/2,ly/2,2*ny-1);

A1 = meanvorticity(lx,ly,nx,ny,sol1.u)
_a = plot(sol1.t,angles,A1',xaxis="t",st=:contourf,color=:bwr,yaxis="<ζ>",label="NL")
savefig(_a,dn*"NL_a_t.png")

A2 = meanvorticity(lx,ly,nx,ny,sol2.u)
_a = plot(sol2.t,angles,A2',xaxis="t",st=:contourf,color=:bwr,yaxis="<ζ>",label="GQL(0)")
savefig(_a,dn*"GQL_"*"$Λ"*"_a_t.png")

A3 = meanvorticity(lx,ly,nx,ny,Λ,sol3.u)
_a = plot(sol3.t,angles,A3',xaxis="t",st=:contourf,color=:bwr,yaxis="<ζ>",label="GCE2(0)")
savefig(_a,dn*"GCE2_"*"$Λ"*"_a_t.png")

## Mean vorticity: t_end
Ajet = real.(ifft(ifftshift(ic_eqm(lx,ly,nx,ny,Ξ)[:,1])))

plot(angles,Ajet,xaxis="θ",yaxis="<ζ>",color=:black,linewidth=2,label="Jet")
plot!(angles,A1[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="NL")
plot!(angles,A2[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="GQL(1)")
plot!(angles,A3[end,:],legend=:bottomright,xaxis="θ",yaxis="<ζ>",linewidth=2,label="GCE2(1)")

plot!(angles,A1[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="NL")
plot!(angles,A2[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="GQL(1)")
plot!(angles,A3[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="GCE2(1)")

## Energy & enstropy

# E1,Z1 = energy(lx,ly,nx,ny,sol1.u);
# _ez = plot(sol1.t,E1,linewidth=2,label="E");
# _ez = plot!(sol1.t,Z1,linewidth=2,legend=:right,yaxis="Energy,Enstrophy",xaxis="Time",label="Z")
# savefig(_ez,dn*"NL_ez_t.png");

# E2,Z2 = energy(lx,ly,nx,ny,sol2.u)
# _ez = plot(sol2.t,E2,linewidth=2,label="E");
# _ez = plot!(sol2.t,Z2,linewidth=2,legend=:right,yaxis="Energy,Enstrophy",xaxis="Time",label="Z")
# savefig(_ez,dn*"GQL_"*"$Λ"*"_ez_t.png");

# E3,Z3 = energy(lx,ly,nx,ny,Λ,sol3.u);
# _ez = plot(sol3.t,E3,linewidth=2,label="E");
# _ez = plot!(sol3.t,Z3,linewidth=2,legend=:right,yaxis="Energy,Enstrophy",xaxis="Time",label="Z")
# savefig(_ez,dn*"GCE2_"*"$Λ"*"_ez_t.png")

## tests
