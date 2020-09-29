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

function exec(lx::Float64,ly::Float64,nx::Int,ny::Int,T::Float64,u0::Array{ComplexF64,2})

    tspan = (0.0,T)
    # u0 = rand(ComplexF64,2*ny-1,nx)
    # u0 = ic_rand(lx,ly,nx,ny)

    A = acoeffs(ly,ny)
    B = bcoeffs(lx,ly,nx,ny)
    Cp,Cm = ccoeffs(lx,ly,nx,ny)
    p = [nx,ny,A,B,Cp,Cm]

    prob = ODEProblem(nl_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=20,dense=false)

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

function exec(lx::Float64,ly::Float64,nx::Int,ny::Int,T::Float64,Ω::Float64,θ::Float64,u0::Array{ComplexF64,2})

    tspan = (0.0,T)
    # u0 = rand(ComplexF64,2*ny-1,nx)
    # u0 = ic_rand(lx,ly,nx,ny)

    A = acoeffs(ly,ny)
    B = bcoeffs(lx,ly,nx,ny,Ω,θ)
    Cp,Cm = ccoeffs(lx,ly,nx,ny)
    p = [nx,ny,A,B,Cp,Cm]

    prob = ODEProblem(nl_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=20,dense=false)

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
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=5)

    return sol

end

function exec(lx::Float64,ly::Float64,nx::Int,ny::Int,T::Float64,Ω::Float64,θ::Float64,νn::Float64,Δθ::Float64,τ::Float64,u0::Array{ComplexF64,2})

    tspan = (0.0,T)

    A = acoeffs(ly,ny,Ω,Δθ,τ)
    B = bcoeffs(lx,ly,nx,ny,Ω,θ,νn,τ)
    Cp,Cm = ccoeffs(lx,ly,nx,ny)
    p = [nx,ny,A,B,Cp,Cm]

    prob = ODEProblem(nl_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=20)

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

function gql(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,T::Float64,u0::Array{ComplexF64,2})

    # u0 = rand(ComplexF64,2*ny-1,nx)
    tspan = (0.0,T)
    # u0 = ic_rand(lx,ly,nx,ny)

    A = acoeffs(ly,ny)
    B = bcoeffs(lx,ly,nx,ny)
    Cp,Cm = ccoeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,A,B,Cp,Cm]

    prob = ODEProblem(gql_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=20,dense=false)

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

function gql(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,T::Float64,Ω::Float64,θ::Float64,νn::Float64,Δθ::Float64,τ::Float64,u0::Array{ComplexF64,2})

    tspan = (0.0,T)

    A = acoeffs(ly,ny,Ω,Δθ,τ)
    B = bcoeffs(lx,ly,nx,ny,Ω,θ,νn,τ)
    Cp,Cm = ccoeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,A,B,Cp,Cm]

    prob = ODEProblem(gql_eqs!,u0,tspan,p)
    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,dense=false,saveat=20)

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
    # poschecktimes = range(1.0,T,step=10.0)
    # condition(u,t,integrator) = t ∈ poschecktimes && !ispositive(u.x[2],nx,ny,Λ)
    # affect!(integrator) = positivity!(integrator.u.x[2],nx,ny,Λ)
    # cb = PresetTimeCallback(poschecktimes,affect!)

    @time sol = solve(prob,RK4(),adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,dense=false)

    return sol

end

function gce2(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,T::Float64,u0f::Array{ComplexF64,2})

    # u0 = rand(ComplexF64,2*ny-1,nx)
    tspan = (0.0,T)
    u0 = ic_cumulants(nx,ny,Λ,u0f)

    A = acoeffs(ly,ny)
    B = bcoeffs(lx,ly,nx,ny)
    Cp,Cm = ccoeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,A,B,Cp,Cm]

    prob = ODEProblem(gce2_eqs!,u0,tspan,p)
    poschecktimes = range(1.0,T,step=20.0)
    condition(u,t,integrator) = t ∈ poschecktimes && !ispositive(u.x[2],nx,ny,Λ)
    affect!(integrator) = positivity!(integrator.u.x[2],nx,ny,Λ)
    cb = PresetTimeCallback(poschecktimes,affect!)

    @time sol = solve(prob,RK4(),callback=cb,adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=20,dense=false)
    # @time sol = solve(prob,RK4(),callback=cb,dt=0.0005,adaptive=false,progress=true,progress_steps=1000,dense=false)

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

    @time sol = solve(prob,RK4(),callback=cb,adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=20)

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
    poschecktimes = [t for t = 10.0:10.0:T]
    condition(u,t,integrator) = t ∈ poschecktimes && !ispositive(u.x[2],nx,ny,Λ)
    affect!(integrator) = positivity!(integrator.u.x[2],nx,ny,Λ)
    cb = PresetTimeCallback(poschecktimes,affect!)

    @time sol = solve(prob,RK4(),callback=cb,adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,saveat=50)

    return sol

end

function gce2(lx::Float64,ly::Float64,nx::Int,ny::Int,Λ::Int,T::Float64,Ω::Float64,θ::Float64,νn::Float64,Δθ::Float64,τ::Float64,u0f::Array{ComplexF64,2})

    tspan = (0.0,T)
    u0 = ic_cumulants(nx,ny,Λ,u0f)
    A = acoeffs(ly,ny,Ω,Δθ,τ)
    B = bcoeffs(lx,ly,nx,ny,Ω,θ,νn,τ)
    Cp,Cm = ccoeffs(lx,ly,nx,ny,Λ)
    p = [nx,ny,Λ,A,B,Cp,Cm]

    prob = ODEProblem(gce2_eqs!,u0,tspan,p)
    # poschecktimes = [t for t = 10.0:10.0:T]
    poschecktimes = range(1.0,T,step=10.0)

    condition(u,t,integrator) = t ∈ poschecktimes && !ispositive(u.x[2],nx,ny,Λ)
    affect!(integrator) = positivity!(integrator.u.x[2],nx,ny,Λ)
    cb = PresetTimeCallback(poschecktimes,affect!)

    @time sol = solve(prob,RK4(),callback=cb,adaptive=true,reltol=1e-6,abstol=1e-6,progress=true,progress_steps=1000,save_start=true,save_everystep=false,dense=false,saveat=20)

    return sol

end
