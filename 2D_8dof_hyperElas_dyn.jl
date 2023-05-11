import ModelingToolkit: Interval, infimum, supremum

using LinearAlgebra
using Lux
using ModelingToolkit
using NeuralPDE
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using Random

@parameters t, x, y
@variables u1(..), u2(..), v1(..), v2(..), TR11(..), TR22(..), TR12(..), TR21(..)
Dy = Differential(y)
Dx = Differential(x)
Dt = Differential(t)

# Material parameters
Gshear = 10 # MPa
Kbulk  = 10 # MPa
lambda = Kbulk - (2.0/3.0)*Gshear
rho    = 0.5  

# Kinematics 
#
# The (plane strain) deformation gradient:
F = [1.0 + Dx(u1(t,x,y)), Dy(u1(t,x,y)), 0,
       Dx(u2(t,x,y)), 1.0 + Dy(u2(t,x,y)), 0,
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
eqs = [Dx(TR11(t,x,y)) + Dy(TR21(t,x,y)) ~ rho*Dt(v1(t,x,y)),
       Dx(TR12(t,x,y)) + Dy(TR22(t,x,y)) ~ rho*Dt(v2(t,x,y)), 
       TR11(t,x,y) ~ TR[1,1],
       TR22(t,x,y) ~ TR[2,2],
       TR21(t,x,y) ~ TR[2,1],
       TR12(t,x,y) ~ TR[1,2]]

λ_u    = 10.0 # Amplification factor for enforcing the disp. boundary conditions
λ_σ    = 1.0 # Amplification factor for enforcing the stress boundary conditions
λ_init = 10.0 # Amplification factor for enforcing the initial conditions

# BCs for simple shear
bcs = [λ_u*u1(t,x,0)   ~ λ_u*0.0,
       λ_u*u2(t,x,0)   ~ λ_u*0.0,
       λ_σ*TR22(t,x,1) ~ λ_σ*0.0, 
       λ_σ*TR12(t,x,1) ~ λ_σ*0.0, 
       λ_σ*TR11(t,0,y) ~ λ_σ*0.0,
       λ_σ*TR21(t,0,y) ~ λ_σ*0.0,
       λ_σ*TR11(t,1,y) ~ λ_σ*0.0,
       λ_σ*TR21(t,1,y) ~ λ_σ*0.0,
       λ_init*v1(0,x,y)  ~ λ_init*2.0*y, # Initial velocity v1
       λ_init*v2(0,x,y)  ~ λ_init*0.0,    # Other ICs are all zero
       λ_init*u1(0,x,y)  ~ λ_init*0.0,    
       λ_init*u2(0,x,y)  ~ λ_init*0.0,
       λ_init*TR11(0,x,y) ~ λ_init*0.0,
       λ_init*TR22(0,x,y) ~ λ_init*0.0,
       λ_init*TR12(0,x,y) ~ λ_init*0.0,
       λ_init*TR21(0,x,y) ~ λ_init*0.0
       ] 

# Space and time domains
domains = [t ∈ Interval(0.0,1.5),
           x ∈ Interval(0.0,1.0),
           y ∈ Interval(0.0,1.0)]

# Neural network
input_ = length(domains) # number of inputs (dimensions)
dofs   = 8               # number of variables expressed by NNs
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

@named pdesystem = PDESystem(eqs,bcs,domains,[t,x,y],[u1(t,x,y),u2(t,x,y), 
                                                      v1(t,x,y),v2(t,x,y),
                                                      TR11(t,x,y), TR22(t,x,y), TR12(t,x,y), TR21(t,x,y)])
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

"""
# Adam is great for getting in the right ballpark of the solution, a "rough pass":
res = Optimization.solve(prob,Adam(1.0e-1); callback = callback, maxiters=500)

# A "refining pass" with Adam:
prob = remake(prob, u0 = res.u)
res = Optimization.solve(prob,Adam(1.0e-2); callback = callback, maxiters=1000)

# Another "refining pass" with Adam:
prob = remake(prob, u0 = res.u)
res = Optimization.solve(prob,Adam(1.0e-3); callback = callback, maxiters=1000)
"""
# LBFGS drills down to finer scales:
#prob = remake(prob, u0 = res.u)
#res = Optimization.solve(prob, NelderMead(); callback = callback, maxiters = 1000)
#res = Optimization.solve(prob, ConjugateGradient(); callback = callback, maxiters = 100)
res = Optimization.solve(prob, LBFGS(); callback = callback, maxiters = 10000)
#res = Optimization.solve(prob, BFGS(); callback = callback, maxiters = 100)

#prob = remake(prob, u0 = res.u)
#res = Optimization.solve(prob,Adam(1.0e-3); callback = callback, maxiters=1000)

using Plots, ColorSchemes, LaTeXStrings

plot(loss_vector, legend=false, yaxis=:log, 
              xlabel="Steps", ylabel="Loss",dpi=600,
              ylimits = (1e-3,1e6))
savefig("PINN_images_8dof_dyn/loss_convergence")


phi = discretization.phi
ts,xs,ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]

minimizers_ = [res.u.depvar[sym_prob.depvars[i]] for i in 1:dofs]

#u_predict  = [[phi[i]([x,y],minimizers_[i])[1] for y in ys  for x in xs] for i in 1:dofs]


meshX = vec((xs'.*ones(size(ys)))')
meshY = vec((ones(size(xs))'.*ys)')


function plot_(phi, minimizers_, ind)
       # Animate
       anim = @animate for (i, t) in enumerate(0:0.0375:1.5)
           @info "Animating frame $i..."
           u1_predict   = [phi[1]([t,x,y],minimizers_[1])[1] for y in ys  for x in xs]
           u2_predict   = [phi[2]([t,x,y],minimizers_[2])[1] for y in ys  for x in xs]
           uind_predict = [phi[ind]([t,x,y],minimizers_[ind])[1] for y in ys  for x in xs]
           defplot_uind = scatter(meshX + 10*u1_predict, meshY + 10*u2_predict, 
                     markersize=2.0,markerstrokewidth=0, xlimits=(-1, 2), ylimits = (-0.1, 1.5),
                     xlabel=L"$x_1, \ \mathrm{mm}.$", ylabel=L"$x_2 \ \mathrm{mm}.$", 
                     marker_z=uind_predict,  c =:coolwarm, aspect_ratio=:equal)
           scatter(defplot_uind, legend=false, dpi=150, size = (800, 400), aspect_ratio=:equal)
       end
       gif(anim,"PINN_images_8dof_dyn/u2_anim.gif", fps=20)
end
   
plot_(phi, minimizers_, 2)
   
