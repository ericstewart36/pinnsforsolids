"""
Code for linear elasticity test problem:
    quasi-static simple shear
    
This code has all the machinery for elastodynamics 
but inertial effects are disabled in  line 275.

- with basic units:
    > Length:  mm
    >   Time:  s
    >   Mass: Mg
  and derived units
    > Pressure: MPa 
    > Force: N
    
    Spring 2023
"""

# Fenics-related packages
from dolfin import *
# Numerical array package
import numpy as np
# Plotting packages
from ufl import sinh
import matplotlib.pyplot as plt
# Current time package
from datetime import datetime


# Set level of detail for log messages (integer)
# 
# Guide:
# CRITICAL  = 50, // errors that may lead to data corruption
# ERROR     = 40, // things that HAVE gone wrong
# WARNING   = 30, // things that MAY go wrong later
# INFO      = 20, // information of general interest (includes solver info)
# PROGRESS  = 16, // what's happening (broadly)
# TRACE     = 13, // what's happening (in detail)
# DBG       = 10  // sundry
#
set_log_level(30)

# The behavior of the form compiler FFC can be adjusted by prescribing
# various parameters. Here, we want to use the UFLACS backend of FFC::
# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"
parameters["form_compiler"]["quadrature_degree"] = 2

'''''''''''''''''''''
DEFINE GEOMETRY
'''''''''''''''''''''

"""
Create mesh and identify the 3D domain and its boundary
"""

# A basic box mesh 
mesh = RectangleMesh(Point(0,0),Point(1, 1),10,10)

x = SpatialCoordinate(mesh) 

#Pick up on the boundary entities of the created mesh
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],0) and on_boundary
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],1) and on_boundary

# Mark boundary subdomains
facets = MeshFunction("size_t", mesh, 1)
facets.set_all(0)
DomainBoundary().mark(facets, 1)  # First, mark all boundaries with common index
# Next mark specific boundaries
Bottom().mark(facets, 2)
Top().mark(facets, 3)

# Define the boundary integration measure "ds".
ds = Measure('ds', domain=mesh, subdomain_data=facets)


'''''''''''''''''''''''''''''''''''''''
MODEL & SIMULATION PARAMETERS
'''''''''''''''''''''''''''''''''''''''

# Choose hyperelastic model (un-comment only one and the corresponding parameters)
############# #################### ###############################################

model = 'NH'   # Neo-Hookean
# Params
Gshear  = Constant(1)    # Shear modulus, MPa
Kbulk   = Constant(1)  # Bulk modulus, MPa
Lambda  = Constant(Kbulk - (2.0/3.0)*Gshear) 

# Mass density
#
# Rubber-like material
rho = Constant(1000) # kg/m^3
# 
# Steel material
#rho = Constant(8e-9) # 8000 kg/m^3 = 8e-9 Mg/mm^3

# Generalized-alpha method parameters
alpha   = Constant(0.0)
gamma   = Constant(0.5+alpha)
beta    = Constant((gamma+0.5)**2/4.)

# Damping coefficient
eta = Constant(0.0) # Constant(1.0e3) 

############# #################### ###############################################

# Simulation time control-related params
t        = 0.0        # start time
Ttot     = 0.25       # total simulation time 
numSteps = 10        # total number of steps
dispTot  = 0.01       # total displacement, mm
#
dt   = Constant(Ttot/numSteps)  # (fixed) step size

# Expression for the initial angular velocity distribution:
omega_0_exp = Expression(("0.0", "0.0", "w0*sin((pi*x[2])/(2*h))"),
                    w0 = 105, pi = np.pi, h = 6, degree=1)

'''''''''''''''''''''
FEM SETUP
'''''''''''''''''''''

# Define function space, both vectorial and scalar
U2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)

TH = MixedElement([U2])  # Mixed element
ME = FunctionSpace(mesh, TH) # Total space for all DOFs

W2 = FunctionSpace(mesh, U2) # Vector space for visulization later
W = FunctionSpace(mesh,P1)   # Scalar space for visulization later

# Define test functions in weak form
u_test = TestFunction(ME)   # Test function

du = TrialFunction(ME) # Trial functions used for automatic differentiation                           

# Define actual functions with the required DOFs
u = Function(ME)

# A copy of functions to store values in last step for time-stepping.
u_old = Function(ME)

# Old functions for storing the velocity and acceleration at prev. step
v_old = Function(W2)
a_old = Function(W2)

# cross product to get initial velocity
#v_0 = cross(omega_0_exp, x)

# set initial velocity condition:
#v_old.assign(project(v_0, W2))

'''''''''''''''''''''
SUBROUTINES
'''''''''''''''''''''

# Quick-calculate sub-routines
    
# Gradient of vector field u   
def pe_grad_vector(u):
    grad_u = grad(u)
    return as_tensor([[grad_u[0,0], grad_u[0,1], 0],
                  [grad_u[1,0], grad_u[1,1], 0],
                  [0, 0, 0]]) 
#
# Gradient of scalar field y
# (just need an extra zero for dimensions to work out)
def pe_grad_scalar(y):
    grad_y = grad(y)
    return as_vector([grad_y[0], grad_y[1], 0.])

    
def sigma(epsilon):
    return Lambda*tr(epsilon)*Identity(3) + 2.0*Gshear*epsilon

# Update formula for acceleration
# a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
def update_a(u, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        beta_ = beta
    else:
        dt_ = float(dt)
        beta_ = float(beta)
    return (u-u_old-dt_*v_old)/beta_/dt_**2 - (1-2*beta_)/2/beta_*a_old

# Update formula for velocity
# v = dt * ((1-gamma)*a0 + gamma*a) + v0
def update_v(a, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        gamma_ = gamma
    else:
        dt_ = float(dt)
        gamma_ = float(gamma)
    return v_old + dt_*((1-gamma_)*a_old + gamma_*a)

def update_fields(u_proj, u_proj_old, v_old, a_old):
    """Update fields at the end of each time step."""

    # Get vectors (references)
    u_vec  = u_proj.vector()
    u_old_vec = u_proj_old.vector()
    v_old_vec = v_old.vector()
    a_old_vec = a_old.vector()

    # use update functions using vector arguments
    a_vec = update_a(u_vec, u_old_vec, v_old_vec, a_old_vec, ufl=False)
    v_vec = update_v(a_vec, u_old_vec, v_old_vec, a_old_vec, ufl=False)

    # Update (v_old <- v, a_old <- a)
    v_old.vector()[:] = v_vec
    a_old.vector()[:] = a_vec
 
# alpha-method averaging function
def avg(x_old, x_new, alpha):
    return alpha*x_old + (1-alpha)*x_new

# function to write results fields to XDMF at time t
def writeResults(t):
    
    # Variable casting and renaming
    _w_1 = u # Get DOFs from last step (or t=0 step)
    
    # Need this to see deformation
    _w_1.rename("Disp.", "Displacement.")
    sig12_v = project(sigma(epsilon)[0,1], W)
    sig12_v.rename("sigma12", "")
    sig11_v = project(sigma(epsilon)[0,0], W)
    sig11_v.rename("sigma11", "")
    sig22_v = project(sigma(epsilon)[1,1], W)
    sig22_v.rename("sigma22", "")
    file_results.write(_w_1, t) 
    file_results.write(sig11_v, t) 
    file_results.write(sig22_v, t) 
    file_results.write(sig12_v, t) 


'''''''''''''''''''''''''''''
KINEMATICS
'''''''''''''''''''''''''''''

# Get acceleration and velocity at end of step
a_new = update_a(u, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)

# get avg fields for generalized-alpha method
u_avg = avg(u_old, u, alpha)
v_avg = avg(v_old, v_new, alpha)

# Deformation measures
epsilon = (1.0/2.0)*(pe_grad_vector(u) + pe_grad_vector(u).T)


'''''''''''''''''''''''''''''
WEAK FORMS
'''''''''''''''''''''''''''''

    
# Weak forms
L_mech = inner(sigma(epsilon), pe_grad_vector(u_test))*dx 
L_mass = rho*inner(a_new, u_test)*dx
L_damp = eta*inner(v_avg, u_test)*dx

# total weak form
L = L_mech #+ L_mass + L_damp

# Automatic differentiation tangent:
a = derivative(L, u, du)
   
'''''''''''''''''''''''
BOUNDARY CONDITIONS
'''''''''''''''''''''''      
dispRamp = Expression(("magnitude*t/Tramp", "0"),
                  magnitude = dispTot, t=0, Tramp = Ttot, degree=1)


# Boundary condition definitions
bcs_1 = DirichletBC(ME, Constant((0,0)), facets, 2)  # u1/u2 fix - Bottom
bcs_2 = DirichletBC(ME, dispRamp, facets, 3)  # u1 fix - Top

# BC set:
bcs = [bcs_1, bcs_2]

'''''''''''''''''''''
    RUN ANALYSIS
'''''''''''''''''''''

# Output file setup
file_results = XDMFFile("results/2D_lineElas_pe_FEniCS.xdmf")
# "Flush_output" permits reading the output during simulation
# (Although this causes a minor performance hit)
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

# Store start time 
startTime = datetime.now()

#Iinitialize a counter for force reporting
iii=0

# initalize output array for tip displacement
totSteps = numSteps+1
timeHist0 = np.zeros(shape=[totSteps])
E_kinetic = np.zeros(shape=[totSteps])
E_elastic = np.zeros(shape=[totSteps])
E_total   = np.zeros(shape=[totSteps])

print("------------------------------------")
print("Simulation Start")
print("------------------------------------")

# Set up the non-linear problem 
stressProblem = NonlinearVariationalProblem(L, u, bcs, J=a)
 
# Set up the non-linear solver
solver  = NonlinearVariationalSolver(stressProblem)

# Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = 'mumps' # 'petsc'   #'gmres'
prm['newton_solver']['absolute_tolerance'] =  1.e-10
prm['newton_solver']['relative_tolerance'] =  1.e-10
prm['newton_solver']['maximum_iterations'] = 30

# Write initial state to XDMF file
writeResults(t=0)

# store non-zero initial KE
E_kinetic[iii] = assemble((1.0/2.0)*rho*dot(v_old, v_old)*dx)
E_total[iii]   = E_kinetic[iii] 

# Time-stepping solution procedure loop
while (round(t + dt, 9) <= Ttot):
    
    # increment time
    t += float(dt)
    dispRamp.t = t
    # increment counter
    iii += 1

    # Solve the problem
    (iter, converged) = solver.solve()
    
    # Write results to file
    writeResults(t)
    
    # Update DOFs for next step
    u_proj = project(u, W2)
    u_proj_old = project(u_old, W2)
    update_fields(u_proj, u_proj_old, v_old, a_old)
    u_old.vector()[:] = u.vector()

    # Print progress of calculation periodically
    if iii%5 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Step: Pull   |   Simulation Time: {}  s  |     Wallclock Time: {}".format(round(t,9), current_time))
        print("Iterations: {}".format(iter))
        print()

# Report elapsed real time for whole analysis
endTime = datetime.now()
elapseTime = endTime - startTime
print("------------------------------------")
print("Elapsed real time:  {}".format(elapseTime))
print("------------------------------------")

# Output data for error calculation in Julia
xs = np.linspace(0, 1, 101)
ys = np.linspace(0, 1, 101)
u1_out = np.zeros(len(xs)*len(ys))
u2_out = np.zeros(len(xs)*len(ys))
i=0
for yi in ys:
    for xi in xs:
        u1_out[i] = u(xi, yi)[0]
        u2_out[i] = u(xi, yi)[1]
        i+= 1
        
np.savetxt("SS_FEniCS_u1.csv", u1_out, delimiter=",")
np.savetxt("SS_FEniCS_u2.csv", u2_out, delimiter=",")