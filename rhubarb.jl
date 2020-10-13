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

""" Set parameters and solve
"""
lx = 4.0*Float64(pi);
ly = 2.0*Float64(pi);
nx = 8;
ny = 10;

Ω = 2.0*Float64(pi)
θ = Float64(pi)/6.0
β = 2.0*Ω*cos(θ)
Ξ = 0.2*Ω
τ = 10.0/Ω
Λ = 0

ζ0 = ic_pert_eqm(lx,ly,nx,ny,Ξ); # one ic for all

sol1 = nl(lx,ly,nx,ny,Ξ,β,τ,ic=ζ0,dt=0.01,t_end=200.0,savefreq=20);
sol2 = gql(lx,ly,nx,ny,Λ,Ξ,β,τ,ic=ζ0,t_end=200.0);
sol3 = gce2(lx,ly,nx,ny,Λ,Ξ,β,τ,ic=ζ0,dt=0.01,t_end=200.0,poscheck=false);

""" Create plots
"""
# plotlyjs();
pyplot();
dn = "tests/7x11/l60j0_2t10/"
mkpath(dn)

## Zonal energy
zones = reshape(["$i" for i = 0:1:nx-1],1,nx);

P1,O1 = zonalenergy(lx,ly,nx,ny,sol1.u);
_p = plot(sol1.t,P1,xaxis=("Time",(-1,51)),yscale=:log10,yaxis=("Energy in Zonal Mode",(1e-12,1e3)),labels=zones,palette=:tab10,legend=:bottomright,linewidth=1.5)
savefig(_p,dn*"NL_em.png");

P2,O2 = zonalenergy(lx,ly,nx,ny,sol2.u);
_p = plot(sol2.t,P2,xaxis=("Time"),yscale=:log10,yaxis=("Energy in Mode",(1e-9,1e3)),labels=zones,palette=:tab10,legend=:best,linewidth=2)
savefig(_p,dn*"GQL_"*"$Λ"*"_em.png")

P3,O3 = zonalenergy(lx,ly,nx,ny,Λ,sol3.u);
_p = plot(sol3.t,P3,xaxis=("Time"),yscale=:log10,yaxis=("Energy in Mode",(1e-9,1e3)),labels=zones,palette=:tab10,legend=:best,linewidth=2)
savefig(_p,dn*"GCE2_"*"$Λ"*"_em.png");

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
_a = plot(sol2.t,angles,A2',xaxis="t",st=:contourf,color=:bwr,yaxis="<ζ>",label="GQL($Λ)")
savefig(_a,dn*"GQL_"*"$Λ"*"_a_t.png")

A3 = meanvorticity(lx,ly,nx,ny,Λ,sol3.u)
_a = plot(sol3.t,angles,A3',xaxis="t",st=:contourf,color=:bwr,yaxis="<ζ>",label="GCE2($Λ)")
savefig(_a,dn*"GCE2_"*"$Λ"*"_a_t.png")

## Mean vorticity: t_end
Ajet = real(ifft(ifftshift(ic_eqm(lx,ly,nx,ny,Ξ)[:,1])))

plot(angles,Ajet,xaxis="θ",yaxis="<ζ>",color=:black,linewidth=2,label="Jet")
plot!(angles,A1[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="NL")
plot!(angles,A2[end,:],xaxis="θ",yaxis="<ζ>",linewidth=2,label="GQL($Λ)")
_zt = plot!(angles,A3[end,:],legend=:bottomright,xaxis="θ",yaxis="<ζ>",linewidth=2,label="GCE2($Λ)")
savefig(_zt,dn*"zt_"*"$Λ"*".png")

## Fourier space energy
mm = LinRange(-(nx-1),nx-1,2*nx-1);
nn = LinRange(-(ny-1),ny-1,2*ny-1);

ef = fourierenergy(lx,ly,nx,ny,sol1.u)
_ef = plot(mm,nn,ef[:,:,begin],st=:contourf,color=:haline,xaxis="kx",yaxis="ky")
_ef = plot(mm,nn,ef[:,:,end],st=:contourf,color=:haline,xaxis="kx",yaxis="ky")
savefig(_a,dn*"NL_ef.png")

ef = fourierenergy(lx,ly,nx,ny,sol2.u)
_ef = plot(mm,nn,ef[:,:,begin],st=:contourf,color=:haline,xaxis="kx",yaxis="ky")
_ef = plot(mm,nn,ef[:,:,end],st=:contourf,color=:haline,xaxis="kx",yaxis="ky")
savefig(_a,dn*"GQL_"*"$Λ"*"_ef.png")

ef = fourierenergy(lx,ly,nx,ny,Λ,sol3.u)
_ef = plot(mm,nn,ef[:,:,begin],st=:contourf,color=:haline,xaxis="kx",yaxis="ky")
_ef = plot(mm,nn,ef[:,:,end],st=:contourf,color=:haline,xaxis="kx",yaxis="ky")
_ef = plot(sol3.t,mm,ef[ny-1,:,:],st=:contourf,color=:haline,xaxis="kx",yaxis="ky")

savefig(_a,dn*"GQL_"*"$Λ"*"_ef.png")

## Zonal velocity
uz = zonalvelocity(lx,ly,nx,ny,sol1.u)
plot(uz[:,:,end],st=:contourf,color=:bwr,xaxis="x",yaxis="y")

uz = zonalvelocity(lx,ly,nx,ny,sol2.u)
plot(uz[:,:,end],st=:contourf,color=:bwr,xaxis="x",yaxis="y")

uz = zonalvelocity(lx,ly,nx,ny,Λ,sol3.u)
plot(uz[:,:,end],st=:contourf,color=:bwr,xaxis="x",yaxis="y")

## Energy & enstropy
e_zon,e_turb = e_lohi(lx,ly,nx,ny,Λ,sol1.u)
_ezt = plot(sol1.t,e_zon,linewidth=2,label="Zonal");
_ezt = plot!(sol1.t,e_turb,linewidth=2,legend=:best,yaxis="Energy",xaxis="Time",label="Turbulence")
plot(e_zon,e_turb)

e_zon,e_turb = e_lohi(lx,ly,nx,ny,Λ,sol2.u)
_ezt = plot(sol2.t,e_zon,linewidth=2,label="Zonal");
_ezt = plot!(sol2.t,e_turb,linewidth=2,legend=:best,yaxis="Energy",xaxis="Time",label="Turbulence")
plot(e_zon,e_turb)

E1,Z1 = energy(lx,ly,nx,ny,sol1.u);
_ez = plot(sol1.t,E1,linewidth=2,label="E");
_ez = plot!(sol1.t,Z1,linewidth=2,legend=:right,yaxis="Energy,Enstrophy",xaxis="Time",label="Z")
# savefig(_ez,dn*"NL_ez_t.png");

# E2,Z2 = energy(lx,ly,nx,ny,sol2.u)
# _ez = plot(sol2.t,E2,linewidth=2,label="E");
# _ez = plot!(sol2.t,Z2,linewidth=2,legend=:right,yaxis="Energy,Enstrophy",xaxis="Time",label="Z")
# savefig(_ez,dn*"GQL_"*"$Λ"*"_ez_t.png");

# E3,Z3 = energy(lx,ly,nx,ny,Λ,sol3.u);
# _ez = plot(sol3.t,E3,linewidth=2,label="E");
# _ez = plot!(sol3.t,Z3,linewidth=2,legend=:right,yaxis="Energy,Enstrophy",xaxis="Time",label="Z")
# savefig(_ez,dn*"GCE2_"*"$Λ"*"_ez_t.png")

## Modal strength
# modes = reshape(["($j,$i)" for j=0:1:nx-1 for i=-(ny-1):1:ny-1],1,nx*(2*ny-1));
#
# M1 = modalstrength(lx,ly,nx,ny,sol1.u);
# _m = plot(sol1.t,M1,labels=modes,legend=:outerright,linewidth=2,xaxis=("Time"),yaxis="Mode Strength")
# savefig(_m,dn*"NL_m_t.png");
#
# M2 = modalstrength(lx,ly,nx,ny,sol2.u);
# _m = plot(sol2.t,M2,labels=modes,legend=:outerright,linewidth=2,xaxis=("Time"),yaxis="Mode Strength")
# savefig(_m,dn*"GQL_"*"$Λ"*"_m_t.png");
#
# M3 = modalstrength(lx,ly,nx,ny,Λ,sol3.u);
# _m = plot(sol3.t,M3,labels=modes,linewidth=2,xaxis=("Time"),yaxis="Mode Strength")
# savefig(_m,dn*"GCE2_"*"$Λ"*"_m_t.png")

## tests
A = acoeffs(ly,ny,Ξ,τ)
B = bcoeffs(lx,ly,nx,ny,β,τ)
Cp,Cm = ccoeffs(lx,ly,nx,ny)
p = [nx,ny,A,B,Cp,Cm]
tspan = (0.0,200.0)

@info "Unoptimized NL equations on $(nx-1)x$(ny-1) grid"
prob = ODEProblem(nl_eqs!,ζ0,tspan,p)
display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))
@info "Removed dζ from NL equations on $(nx-1)x$(ny-1) grid"
prob = ODEProblem(nl_eqs2!,ζ0,tspan,p)
display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))
@info "Running that using Tsit5"
prob = ODEProblem(nl_eqs2!,ζ0,tspan,p)
display(@benchmark solve(prob,Tsit5(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))

A = acoeffs(ly,ny,Ξ,τ)
B = bcoeffs(lx,ly,nx,ny,β,τ)
Cp,Cm = ccoeffs(lx,ly,nx,ny)
p = [nx,ny,Λ,A,B,Cp,Cm]
tspan = (0.0,200.0)

Λ = nx - 2
@info "Unoptimized GQL equations on $(nx-1)x$(ny-1) grid"
prob = ODEProblem(gql_eqs!,ζ0,tspan,p)
display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))
@info "Removed dζ from GQL equations on $(nx-1)x$(ny-1) grid"
prob = ODEProblem(gql_eqs2!,ζ0,tspan,p)
display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))
@info "Running that using Tsit5"
prob = ODEProblem(gql_eqs2!,ζ0,tspan,p)
display(@benchmark solve(prob,Tsit5(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))

@info "Unoptimized GCE2($Λ) equations on $(nx-1)x$(ny-1) grid"
prob = ODEProblem(gce2_eqs!,ic_cumulants(nx,ny,Λ,ζ0),tspan,p)
display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))
@info "Removed dζ from GCE2($Λ) equations on $(nx-1)x$(ny-1) grid"
prob = ODEProblem(gce2_eqs2!,ic_cumulants(nx,ny,Λ,ζ0),tspan,p)
display(@benchmark solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20))
sol3 = solve(prob,RK4(),dt=0.01,adaptive=false,progress=true,progress_steps=10000,save_start=false,save_everystep=false,saveat=20)
