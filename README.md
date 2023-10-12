***Physics-informed neural networks (PINNs) versus finite elements for solving PDEs in continuum mechanics of solids***.

# Abstract:

In the last three years in the continuum mechanics of solids literature, physics-informed neural networks (PINNs) which use neural networks for numerical solution of partial differential equations (PDEs) have mounted a challenge to the de facto ruler of PDE solvers in that field: the finite element method (FEM). Such PINNS have intrinsic advantages over FEM in that (i) they are mesh-free and (ii) they do not have an explicit time discretization; however PINNs as a numerical method for solving PDEs are in their infancy and are notoriously difficult to work with. In this project, I use the modern language Julia to construct two PINNS for example problems in (i) linear elasticity and (i) finite deformation hyperelasticity, documenting my challenges and strategies developed along the way. I compare the PINN results to FEM results both for the purposes of validation and of performance comparisons between the two methods. The PINN and FEM codes for this project are publicly available on GitHub, along with a list of tips and tricks for constructing PINNs that I accumulated during my project development.


# Some tips and tricks for PINNs

1.  **BC enforcement:** Unlike FEM, PINNs only ``weakly'' enforce the boundary conditions via the loss function --- that is, they permit solutions which do not completely satisfy the applied boundary conditions. If you have trouble with BC enforcement, you can use the augmented loss function scheme (4.10) from my report to improve the enforcement of the BCs by setting $\lambda_{\text{\tiny BCs}} > \lambda_{\text{\tiny PDE}}$.


 - The paper [Rao et al. (2021)](https://ascelibrary.org/doi/full/10.1061/%28ASCE%29EM.1943-7889.0001947) presents a scheme for forcible boundary condition enforcement using a composition of three functions: one which satisfies the boundary conditions precisely, one which measures the distance to the nearest boundary condition "edge", and one which represents the solution within the domain. Implementing this approach in `NeuralPDE.jl` could be a nice future project.

2. **Low-order derivatives:** I found that reformulating my problem in terms of lower-order derivatives --- even at the cost of extra governing equations and degrees of freedom --- was *essential* to getting accurate results from my PINN. This observation has been made before in the literature for TensorFlow-based PINNs by [Rao et al. (2021)](https://ascelibrary.org/doi/full/10.1061/%28ASCE%29EM.1943-7889.0001947). 

3.  **Neural network architecture:** I found that solving PDEs on a unit square does not require very deep or wide neural networks --- 2 hidden layers with 5 hidden nodes each and hyperbolic tangent activation was sufficient. In general, the optimization process and problem formulation matter much more than neural network architecture.

4. **Optimization algorithms:** Adam and BFGS/LBFGS seem to work the best for training PINNS --- Adam for a fast "burn-in" process and BFGS/LBFGS to drill down to finer scales.

5. **Material parameters:** I found that using physical values of the material parameters (e.g. steel has  $G=210,000$ MPa and $K = 245,000$ MPa) led to poor convergence behavior since the different components of the loss function had different orders of magnitude. For best convergence in physical problems, it is essential to devise a unit system and normalization scheme which renders the different components of the loss functions similar orders of magnitude.

- In this work I sidestep the issue of material parameter scaling by simply using unit material parameters rather than physically representative values. However, for PINNs to be useful in engineering they must be able to accommodate physical values of material parameters.

