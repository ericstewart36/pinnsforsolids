import ModelingToolkit: Interval, infimum, supremum

using LinearAlgebra
using Lux
using ModelingToolkit
using NeuralPDE
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using Random

@parameters x, y
@variables u1(..), u2(..), σ11(..), σ22(..), σ12(..)
Dy = Differential(y)
Dx = Differential(x)

# Material parameters
Gshear = 1 # MPa
Kbulk  = 1 # MPa
lambda = Kbulk - (2.0/3.0)*Gshear

# Governing equations for force balance, constitutive relations
eqs = [Dx(σ11(x,y)) + Dy(σ12(x,y)) ~ 0.0, 
       Dx(σ12(x,y)) + Dy(σ22(x,y)) ~ 0.0,
       σ11(x,y) ~ lambda*(Dx(u1(x,y)) + Dy(u2(x,y))) + 2.0*Gshear*Dx(u1(x,y)),
       σ22(x,y) ~ lambda*(Dx(u1(x,y)) + Dy(u2(x,y))) + 2.0*Gshear*Dy(u2(x,y)),
       σ12(x,y) ~ Gshear*(Dy(u1(x,y)) + Dx(u2(x,y))) ]

λ_u = 1.0 # Amplification factor for enforcing the disp. boundary conditions
λ_σ = 1.0 # Amplification factor for enforcing the stress boundary conditions

# BCs for simple shear
bcs = [λ_u*u1(x,0)  ~ λ_u*0.0,
       λ_u*u2(x,0)  ~ λ_u*0.0,
       λ_u*u1(x,1)  ~ λ_u*0.01,
       λ_u*u2(x,1)  ~ λ_u*0.0,
       λ_σ*σ11(0,y) ~ λ_σ*0.0,
       λ_σ*σ12(0,y) ~ λ_σ*0.0,
       λ_σ*σ11(1,y) ~ λ_σ*0.0,
       λ_σ*σ12(1,y) ~ λ_σ*0.0
       ] 

# Space and time domains
domains = [x ∈ Interval(0.0,1.0),
           y ∈ Interval(0.0,1.0)]

# Neural network
input_ = length(domains) # number of inputs (dimensions)
dofs   = 5               # number of variables expressed by NNs
n      = 5              # dimension of hidden NN layer
chain =[Lux.Chain(Dense(input_,n),
                  Dense(n,     n, Lux.tanh_fast),
                  #Dense(n,     n, Lux.tanh_fast),
                  #Dense(n,     n, Lux.tanh_fast),
                  #Dense(n,     n, Lux.tanh_fast),
                  #Dense(n,     n, Lux.tanh_fast),
                  Dense(n,1)) for _ in 1:dofs]

strategy = QuadratureTraining()
#strategy = QuasiRandomTraining(100)
#strategy = GridTraining(0.05)

"""
discretization = PhysicsInformedNN(chain, strategy, 
                                    adaptive_loss = GradientScaleAdaptiveLoss(reweight_every=10;
                                    weight_change_inertia = 0.9,
                                    pde_loss_weights = 1,
                                    bc_loss_weights = 1,
                                    additional_loss_weights = 1))
"""
discretization = PhysicsInformedNN(chain, strategy)

@named pdesystem = PDESystem(eqs,bcs,domains,[x,y],[u1(x, y),u2(x, y), σ11(x,y), σ22(x,y), σ12(x,y)])
prob = discretize(pdesystem,discretization)
sym_prob = symbolic_discretize(pdesystem,discretization)

pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions

loss_vector = Vector{Float64}()

callback = function (p, l)
    push!(loss_vector, l)
    stepnum = length(loss_vector)
    @info "Step number $stepnum."
    println("loss: ", l)
    println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p), bcs_inner_loss_functions))
    return false
end

# Adam is great for getting in the right ballpark of the solution, a "rough pass":
res = Optimization.solve(prob,Adam(1.0e-3); callback = callback, maxiters=2000)

# A "refining pass" with Adam:
prob = remake(prob, u0 = res.u)
res = Optimization.solve(prob,Adam(1.0e-4); callback = callback, maxiters=2000)

# LBFGS drills down to finer scales:
prob = remake(prob, u0 = res.u)
res = Optimization.solve(prob, LBFGS(); callback = callback, maxiters = 5000)

using Plots, ColorSchemes, LaTeXStrings

plot(loss_vector, legend=false, yaxis=:log, 
              xlabel="Steps", ylabel="Loss",dpi=600,
              ylimits = (1e-6,1e2))
savefig("loss_convergence")


phi = discretization.phi
xs,ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]

minimizers_ = [res.u.depvar[sym_prob.depvars[i]] for i in 1:dofs]

u_predict  = [[phi[i]([x,y],minimizers_[i])[1] for y in ys  for x in xs] for i in 1:dofs]


meshX = vec((xs'.*ones(size(ys)))')
meshY = vec((ones(size(xs))'.*ys)')

# Draw the reference geometry
refplot    = scatter(meshX, meshY, title = L"\mathrm{Reference \ body.}", mc=:gray, 
                  markersize=1.0,markerstrokewidth=0, xlimits=(-0.1, 1.2), ylimits = (-0.1, 1.1), 
                  xlabel=L"$X_1, \ \mathrm{mm}.$", ylabel=L"$X_2, \ \mathrm{mm}.$", aspect_ratio=:equal) 
scatter(refplot, legend=false, size = (400, 300), dpi=600, aspect_ratio=:equal)
savefig("2d_linElas_ref_5dof")

# Draw contours of dofs on deformed geometry
defplot_u1 = scatter(meshX + 10*u_predict[1], meshY + 10*u_predict[2], 
                     title = L"\mathrm{Contours \ of \ } u_1(x,y) \mathrm{\ (mm).}",
                     markersize=2.0,markerstrokewidth=0, xlimits=(-0.1, 1.2), ylimits = (-0.1, 1.1),
                     xlabel=L"$x_1, \ \mathrm{mm}.$", ylabel=L"$x_2 \ \mathrm{mm}.$", 
                     marker_z=u_predict[1],  c =:coolwarm, aspect_ratio=:equal)
scatter(defplot_u1, legend=false, dpi=600, size = (800, 400), aspect_ratio=:equal)
savefig("2d_linElas_def_u1_5dof")

defplot_u2 = scatter(meshX + 10*u_predict[1], meshY + 10*u_predict[2],
                     title = L"\mathrm{Contours \ of \ } u_2(x,y) \mathrm{\ (mm).}",
                     markersize=2.0,markerstrokewidth=0, xlimits=(-0.1, 1.2), ylimits = (-0.1, 1.1),
                     xlabel=L"$x_1, \ \mathrm{mm}.$", ylabel=L"$x_2 \ \mathrm{mm}.$", 
                     marker_z=u_predict[2], c =:coolwarm )
scatter(defplot_u2, legend=false, dpi=600, size = (800, 400), aspect_ratio=:equal)
savefig("2d_linElas_def_u2_5dof")

defplot_σ11 = scatter(meshX + 10*u_predict[1], meshY + 10*u_predict[2],
                     title = L"\mathrm{Contours \ of \ } \sigma_{11}(x,y) \mathrm{\ (mm).}",
                     markersize=2.0,markerstrokewidth=0, xlimits=(-0.1, 1.2), ylimits = (-0.1, 1.1),
                     xlabel=L"$x_1, \ \mathrm{mm}.$", ylabel=L"$x_2 \ \mathrm{mm}.$", 
                     marker_z=u_predict[3], c =:coolwarm )
scatter(defplot_σ11, legend=false, dpi=600, size = (800, 400), aspect_ratio=:equal)
savefig("2d_linElas_def_sigma11_5dof")

defplot_σ22 = scatter(meshX + 10*u_predict[1], meshY + 10*u_predict[2],
                     title = L"\mathrm{Contours \ of \ } \sigma_{22}(x,y) \mathrm{\ (mm).}",
                     markersize=2.0,markerstrokewidth=0, xlimits=(-0.1, 1.2), ylimits = (-0.1, 1.1),
                     xlabel=L"$x_1, \ \mathrm{mm}.$", ylabel=L"$x_2 \ \mathrm{mm}.$", 
                     marker_z=u_predict[4], c =:coolwarm )
scatter(defplot_σ22, legend=false, dpi=600, size = (800, 400), aspect_ratio=:equal)
savefig("2d_linElas_def_sigma22_5dof")

defplot_σ12 = scatter(meshX + 10*u_predict[1], meshY + 10*u_predict[2],
                     title = L"\mathrm{Contours \ of \ } \sigma_{12}(x,y) \mathrm{\ (mm).}",
                     markersize=2.0,markerstrokewidth=0, xlimits=(-0.1, 1.2), ylimits = (-0.1, 1.1),
                     xlabel=L"$x_1, \ \mathrm{mm}.$", ylabel=L"$x_2 \ \mathrm{mm}.$", 
                     marker_z=u_predict[5], c =:coolwarm )
scatter(defplot_σ12, legend=false, dpi=600, size = (800, 400), aspect_ratio=:equal)
savefig("2d_linElas_def_sigma12_5dof")

using CSV, DataFrames

FEniCS_u1_data = CSV.read("SS_FEniCS_u1.csv", DataFrame, header=false)
FEniCS_u2_data = CSV.read("SS_FEniCS_u2.csv", DataFrame, header=false)

FEniCS_u1_arr = Matrix(FEniCS_u1_data)
FEniCS_u2_arr = Matrix(FEniCS_u2_data)

FEniCS_u1_err = abs.(u_predict[1] .- FEniCS_u1_arr)
FEniCS_u2_err = abs.(u_predict[2] .- FEniCS_u2_arr)

# Draw contours of dofs on deformed geometry
FEniCSplot_u1 = scatter(meshX + 10*FEniCS_u1_arr, meshY + 10*FEniCS_u2_arr, 
                     title = L"\mathrm{Error \ in \ } u_1(x,y)",
                     markersize=2.0,markerstrokewidth=0, xlimits=(-0.1, 1.2), ylimits = (-0.1, 1.1),
                     xlabel=L"$x_1, \ \mathrm{mm}.$", ylabel=L"$x_2 \ \mathrm{mm}.$", 
                     marker_z=FEniCS_u1_err,  c =:coolwarm, aspect_ratio=:equal)
scatter(FEniCSplot_u1, legend=false, dpi=600, size = (800, 400), aspect_ratio=:equal)
savefig("SS_FEniCS_u1_err.png")

# Draw contours of dofs on deformed geometry
FEniCSplot_u2 = scatter(meshX + 10*FEniCS_u1_arr, meshY + 10*FEniCS_u2_arr, 
                     title = L"\mathrm{Error \ in \ } u_2(x,y)",
                     markersize=2.0,markerstrokewidth=0, xlimits=(-0.1, 1.2), ylimits = (-0.1, 1.1),
                     xlabel=L"$x_1, \ \mathrm{mm}.$", ylabel=L"$x_2 \ \mathrm{mm}.$", 
                     marker_z=FEniCS_u2_err,  c =:coolwarm, aspect_ratio=:equal)
scatter(FEniCSplot_u2, legend=false, dpi=600, size = (800, 400), aspect_ratio=:equal)
savefig("SS_FEniCS_u2_err.png")

u1_err_rms = sqrt(sum(FEniCS_u1_err.^2)/length(FEniCS_u1_err))
u2_err_rms = sqrt(sum(FEniCS_u2_err.^2)/length(FEniCS_u2_err))
