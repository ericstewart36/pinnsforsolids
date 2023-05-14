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
@variables u1(..), u2(..), TR11(..), TR22(..), TR12(..), TR21(..)
Dy = Differential(y)
Dx = Differential(x)

# Material parameters
Gshear = 1 # MPa
Kbulk  = 1 # MPa
lambda = Kbulk - (2.0/3.0)*Gshear


# Kinematics 
#
# The (plane strain) deformation gradient:
F = [1.0 + Dx(u1(x,y)), Dy(u1(x,y)), 0,
       Dx(u2(x,y)), 1.0 + Dy(u2(x,y)), 0,
 		0, 0, 1.0]
# F as a matrix
Fmat = reshape(F, (3,3))'
#
# F^{-T}
Fit = inv(Fmat')
#
# J = det(F)
J = det(Fmat)

# Piola Stress
TR = Gshear*(Fmat - Fit) + Kbulk*(J-1)*Fit

# Governing eqs
eqs = [Dx(TR11(x,y)) + Dy(TR21(x,y)) ~ 0,
       Dx(TR12(x,y)) + Dy(TR22(x,y)) ~ 0, 
       TR11(x,y) ~ TR[1,1],
       TR22(x,y) ~ TR[2,2],
       TR21(x,y) ~ TR[1,2],
       TR12(x,y) ~ TR[2,1]]

λ_u = 10.0 # Amplification factor for enforcing the disp. boundary conditions
λ_σ = 10.0 # Amplification factor for enforcing the stress boundary conditions

# BCs for simple shear
bcs = [λ_u*u1(x,0)  ~ λ_u*0.0,
       λ_u*u2(x,0)  ~ λ_u*0.0,
       λ_u*u1(x,1)  ~ λ_u*0.0, 
       λ_u*u2(x,1)  ~ λ_u*1.0, # Beeeeg stretch (λ=2)
       λ_σ*TR11(0,y) ~ λ_σ*0.0,
       λ_σ*TR21(0,y) ~ λ_σ*0.0,
       λ_σ*TR11(1,y) ~ λ_σ*0.0,
       λ_σ*TR21(1,y) ~ λ_σ*0.0
       ] 

# Space and time domains
domains = [x ∈ Interval(0.0,1.0),
           y ∈ Interval(0.0,1.0)]

# Neural network
input_ = length(domains) # number of inputs (dimensions)
dofs   = 6               # number of variables expressed by NNs
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

@named pdesystem = PDESystem(eqs,bcs,domains,[x,y],[u1(x, y),u2(x, y), TR11(x,y), TR22(x,y), TR12(x,y), TR21(x,y)])
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

# Here LBFGS does a great job, although each step is quite computationally expensive.
# We get acceptable results in a reasonable time frame for 500 steps.
res = Optimization.solve(prob, LBFGS(); callback = callback, maxiters = 500)

using Plots, ColorSchemes, LaTeXStrings

plot(loss_vector, legend=false, yaxis=:log, 
              xlabel="Steps", ylabel="Loss",dpi=600,
              ylimits = (1e0,1e4))
savefig("loss_convergence")


phi = discretization.phi
xs,ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]

minimizers_ = [res.u.depvar[sym_prob.depvars[i]] for i in 1:dofs]

u_predict  = [[phi[i]([x,y],minimizers_[i])[1] for y in ys  for x in xs] for i in 1:dofs]


meshX = vec((xs'.*ones(size(ys)))')
meshY = vec((ones(size(xs))'.*ys)')

# Draw the reference geometry
refplot   = scatter(meshX, meshY, title = L"\mathrm{Reference \ body.}", mc=:gray, 
                  markersize=2.0,markerstrokewidth=0, xlimits=(-0.1, 1.1), ylimits = (-0.1, 2.1), 
                  xlabel=L"$X_1, \ \mathrm{mm}.$", ylabel=L"$X_2, \ \mathrm{mm}.$", aspect_ratio=:equal) 
scatter(refplot, legend=false, size = (800, 400), dpi=600, aspect_ratio=:equal)
savefig("2d_hyperElas_refplot")


# Draw the reference geometry
refplot   = scatter(meshX +u_predict[1], meshY + u_predict[2], title = L"\mathrm{Deformed \ body.}", mc=:gray, 
                  markersize=2.0,markerstrokewidth=0, xlimits=(-0.1, 1.1), ylimits = (-0.1, 2.1), 
                  xlabel=L"$X_1, \ \mathrm{mm}.$", ylabel=L"$X_2, \ \mathrm{mm}.$", aspect_ratio=:equal) 
scatter(refplot, legend=false, size = (800, 400), dpi=600, aspect_ratio=:equal)
savefig("2d_hyperElas_defplot")

# Draw contours of dofs on deformed geometry
defplot_u1 = scatter(meshX + u_predict[1], meshY + u_predict[2], 
                     title = L"\mathrm{Contours \ of \ } u_1(x,y) \mathrm{\ (mm).}",
                     markersize=2.0,markerstrokewidth=0, xlimits=(-0.1, 1.2), ylimits = (-0.1, 2.1),
                     xlabel=L"$x_1, \ \mathrm{mm}.$", ylabel=L"$x_2 \ \mathrm{mm}.$", 
                     marker_z=u_predict[1],  c =:coolwarm, aspect_ratio=:equal)
scatter(defplot_u1, legend=false, dpi=600, size = (800, 400), aspect_ratio=:equal)
savefig("2d_hypElas_def_u1_6dof")

defplot_u2 = scatter(meshX + u_predict[1], meshY + u_predict[2],
                     title = L"\mathrm{Contours \ of \ } u_2(x,y) \mathrm{\ (mm).}",
                     markersize=2.0,markerstrokewidth=0, xlimits=(-0.1, 1.2), ylimits = (-0.1, 2.1),
                     xlabel=L"$x_1, \ \mathrm{mm}.$", ylabel=L"$x_2 \ \mathrm{mm}.$", 
                     marker_z=u_predict[2], c =:coolwarm )
scatter(defplot_u2, legend=false, dpi=600, size = (800, 400), aspect_ratio=:equal)
savefig("2d_hypElas_def_u2_6dof")

defplot_TR11 = scatter(meshX + u_predict[1], meshY + u_predict[2],
                     title = L"\mathrm{Contours \ of \ } T_{{R},11}(x,y) \mathrm{\ (mm).}",
                     markersize=2.0,markerstrokewidth=0, xlimits=(-0.1, 1.2), ylimits = (-0.1, 2.1),
                     xlabel=L"$x_1, \ \mathrm{mm}.$", ylabel=L"$x_2 \ \mathrm{mm}.$", 
                     marker_z=u_predict[3], c =:coolwarm )
scatter(defplot_TR11, legend=false, dpi=600, size = (800, 400), aspect_ratio=:equal)
savefig("2d_hypElas_def_TR11_6dof")


defplot_TR22 = scatter(meshX + u_predict[1], meshY + u_predict[2],
                     title = L"\mathrm{Contours \ of \ } T_{{R},22}(x,y) \mathrm{\ (mm).}",
                     markersize=2.0,markerstrokewidth=0, xlimits=(-0.1, 1.2), ylimits = (-0.1, 2.1),
                     xlabel=L"$x_1, \ \mathrm{mm}.$", ylabel=L"$x_2 \ \mathrm{mm}.$", 
                     marker_z=u_predict[4], c =:coolwarm )
scatter(defplot_TR22, legend=false, dpi=600, size = (800, 400), aspect_ratio=:equal)
savefig("2d_hypElas_def_TR22_6dof")

defplot_TR12 = scatter(meshX + u_predict[1], meshY + u_predict[2],
                     title = L"\mathrm{Contours \ of \ } T_{{R},12}(x,y) \mathrm{\ (mm).}",
                     markersize=2.0,markerstrokewidth=0, xlimits=(-0.1, 1.2), ylimits = (-0.1, 2.1),
                     xlabel=L"$x_1, \ \mathrm{mm}.$", ylabel=L"$x_2 \ \mathrm{mm}.$", 
                     marker_z=u_predict[5], c =:coolwarm )
scatter(defplot_TR12, legend=false, dpi=600, size = (800, 400), aspect_ratio=:equal)
savefig("2d_hypElas_def_TR12_6dof")

defplot_TR21 = scatter(meshX + u_predict[1], meshY + u_predict[2],
                     title = L"\mathrm{Contours \ of \ } T_{{R},21}(x,y) \mathrm{\ (mm).}",
                     markersize=2.0,markerstrokewidth=0, xlimits=(-0.1, 1.2), ylimits = (-0.1, 2.1),
                     xlabel=L"$x_1, \ \mathrm{mm}.$", ylabel=L"$x_2 \ \mathrm{mm}.$", 
                     marker_z=u_predict[6], c =:coolwarm )
scatter(defplot_TR21, legend=false, dpi=600, size = (800, 400), aspect_ratio=:equal)
savefig("2d_hypElas_def_TR21_6dof")
