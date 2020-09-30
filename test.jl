# type stability checks
# opt_eqs()
# @btime nl_coeffs(lx,ly,nx,ny)
# du = similar(u0)
# @code_warntype nl_eqs!(du,u0,p,tspan)
# @code_warntype gql_eqs!(du,u0,p,tspan)
# @code_warntype gce2_eqs!(du,u0,p,tspan)

function plot4time(var,fn::String,lx::Float64,ly::Float64,nx::Int,ny::Int)

    x = LinRange(-lx,lx,2*nx-1)
    y = LinRange(-ly,ly,2*ny-1)

    anim = @animate for i ∈ 1:length(var[1,1,:])
        Plots.plot(x,y,var[:,:,i],st=:contourf,color=:bwr,xaxis="x",yaxis="y",title=(i-1)*50,aspect=:equal)
    end
    gif(anim, fn, fps = 0.5)
    # return nothing
end

function opt_eqs()

    samples = 7
    timings = zeros(samples)
    for i in 1:1:samples

        nx = i + 1
        ny = i + 1

        println("Solving Nx2N system with N = ", nx)
        u0 = randn(ComplexF64,2*ny-1,nx)
        tspan = (0.0,100.0)
        Cp,Cm = nl_coeffs(lx,ly,nx,ny)
        p = [nx,ny,Cp,Cm]
        prob = ODEProblem(nl_eqs!,u0,tspan,p)
        timings[i] = @elapsed solve(prob,RK4(),adaptive=true,progress=true,save_start=false,save_everystep=false)

    end

    dims = [i + 1 for i in 1:samples]
    Plots.plot(dims,timings,scale=:log,xaxis="N",yaxis="T",markershape = :square,legend=false)

end
