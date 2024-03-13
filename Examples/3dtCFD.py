'''
Convergence test for a mixed scheme for the NS equations
Steady
Domain is (-1,1)^2
Alfeld splits (barycentric refinements done with GMSH, see below *)
Manufactured smooth solutions

strong primal form: 

gamma u - div(2mu*eps(u)) + grad(u)*u + grad(p) =  g
                                              div(u) = 0

Pure Dirichlet conditions 
Lagrange multiplier to fix the average of p

strong mixed form in terms of (u,t,sigma)

grad(u) = t
gamma u - div(sigma) + 0.5*t*u - g = 0
2*mu*t_sym - 0.5*dev(u otimes u) = dev(sigma)
avg(tr(2*sigma+u otimes u)) = 0

RT(k) elements for the rows of sigma and DG(k) elements for everyone else

(*) Barycentric refinement: 

gmsh generateUnitSquares.geo - 
then for each file: gmsh -barycentric_refine file.msh
then for each file: meshio-convert file.msh file.xml -z -p (remove options if in 3D)

'''

from fenics import *
import sympy2fenics as sf
import sympy as sp
import numpy as np 
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4
list_linear_solver_methods()

fileO = XDMFFile("outputs/out-NS.xdmf")
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True

# ************ Time units *********************** #

time = 0.0; Tfinal = 0.5;  dt = 0.001; 
inc = 0;   frequency = Tfinal/dt;

# ****** Constant coefficients ****** #


#gamma = Constant(1.0e-3)
pi = 3.1415926535897932384626
c  = Constant(0.1)
g  = Constant((0.0,0.0,-9.8*(1+0.1*c)))
f     = Constant((0.,0.,0.))
Id = Constant(((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)))
u_D = Constant((0.,0.,0.))
# *********** Variable coefficients ********** #

mu   = Constant(0.01) #lambda phi: exp(-phi)
#x, y, z = sp.symbols('x[0] x[1] x[2]')
def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

# ******* Exact solutions for error analysis ****** #
#u_str = '(4*y**2*(x**2-1)**2*(y**2-1),-4*x**2*(y**2-1)**2*(x**2-1))'
p_str = 'sin(pi*x)*sin(pi*y)*sin(pi*z)'
#u_str = '(  4*x*y*z*(z-1)*(y-1)*(y-z)*pow((x-1),2.)   ,  -4*x*pow(y,2.)*z*pow( (y-1),2.)*(z-1)*(x-1)*(x-z)    ,  4*x*y*pow(z,2.)*pow((z-1),2.)*(y-1)*(x-1)*pow((x-y),2.)  )'
#u_str = '(  4*x*y*z*(z-1)*(y-1)*(y-z)*pow((x-1),2.)   ,  -4*x*pow(y,2.)*z*pow( (y-1),2.)*(z-1)*(x-1)*(x-z)    ,  4*x*y*pow(z,2.)*pow((z-1),2.)*(y-1)*(x-1)*pow((x-y),2.)  )'
u_str = '(  sin(pi*x)*cos(pi*y)*cos(pi*z)   , -2.*cos(pi*x)*sin(pi*y)*cos(pi*z)     ,  cos(pi*x)*cos(pi*y)*sin(pi*z)    )'

#class MyExpression(UserExpression):
#  def eval(self, value, x):
#    if x[0] <= 0+ DOLFIN_EPS:
#      value[0] = 10.0
#    else:
#      value[0] = 0.0

#  def value_shape(self):
#    return (3,)

#u_D = MyExpression()

#================================================================
# only for rates
#================================================================
hvec = []; nvec = []; erru = []; rateu = []; errp = []; ratep = [];
errt = []; ratet = []; errsigma = []; ratesigma = []
errphi = []; ratephi = []; errsigmaj = []; ratesigmaj = []; 
errtj = []; ratetj = []; 

rateu.append(0.0); ratet.append(0.0); ratephi.append(0.0);
ratep.append(0.0); ratetj.append(0.0); 
ratesigma.append(0.0); ratesigmaj.append(0.0); 
#================================================================
nk=2
meshname="meshes/boxes3D.xml"
mesh = Mesh(meshname)
#mesh = UnitCubeMesh(10, 10, 10)
nn   = FacetNormal(mesh)
# rates #================================================================
hvec.append(mesh.hmax())
#================================================================

#================================================================
    # Exact solutions settings
#================================================================
    
#u_ex     = Expression(str2exp(u_str), degree=7, domain=mesh)
#p_ex     = Expression(str2exp(p_str), degree=6, domain=mesh)
#t_ex     = grad(u_ex)
#sigma_ex = 2.*mu*sym(t_ex) - 0.5*outer(u_ex,u_ex) - p_ex*Id
#rhs_ex   = gamma*u_ex - div(sigma_ex) + 0.5*t_ex*u_ex -g
#================================================================
    # Exact solutions settings
#================================================================
# *********** Finite Element spaces ************* #
u_ex     = Expression(str2exp(u_str), degree=7, domain=mesh)
deg = 1
Pkv = VectorElement('DG', mesh.ufl_cell(), deg)
Pk  = FiniteElement('DG', mesh.ufl_cell(), deg)
RTv = FiniteElement('RT', mesh.ufl_cell(), deg+1)
R0  = FiniteElement('R', mesh.ufl_cell(), 0)
Hh  = FunctionSpace(mesh, MixedElement([Pkv,Pk,Pk,Pk,Pk,Pk,Pk,Pk,Pk,RTv,RTv,RTv,R0]))
Ph  = FunctionSpace(mesh,'CG',1)
print (" ****** Total DoF = ", Hh.dim())
#================================================================
    # *********** Trial and test functions ********** #
#================================================================s
Utrial = TrialFunction(Hh)
Usol   = Function(Hh)
u, t11, t12, t13,t21,t22,t23,t31,t32, Rsig1, Rsig2,Rsig3, xi= split(Usol)
v, s11, s12, s13,s21,s22,s23,s31,s32, Rtau1, Rtau2,Rtau3, ze= TestFunctions(Hh)
#v, s11, s12, s21, Rtau1, Rtau2, ze= TestFunctions(Hh)

t = as_tensor( ( (t11,t12,t13) , (t21,t22,t23) , (t31,t32,-t22-t11) ) )
s = as_tensor( ( (s11,s12,s13) , (s21,s22,s23) , (s31,s32,-s22-s11) ) )
    
sigma = as_tensor((Rsig1,Rsig2,Rsig3))
tau   = as_tensor((Rtau1,Rtau2,Rtau3))

# ********** Initial conditions ******** #

uold = Function(Hh.sub(0).collapse())
#perth  = u_ex
init = u_ex
#init  = Expression("(x[1]>-80 && x[1]<-40) ? i1 : 0.", i1=perth, degree=2)
uold  = interpolate(init,Hh.sub(0).collapse())
#gamma = Constant(1.)#interpolate(1./(1.+0.25*(perth/0.001-0.999)),Rh)

    # ********** Boundary conditions ******** #

    # All Dirichlet BCs become natural in this mixed form
    
    # *************** Variational forms ***************** #
        # flow equations
    
#BigA = 1./dt*( (u[0]+ uold[0])*v[0] + (u[1]+ uold[1])*v[1] + (u[2]+ uold[2])*v[2] )*dx  - dot(div(sigma),v)*dx + 0.5*dot(t*u,v)*dx  # + gamma*dot(u,v)*dx - dot(div(sigma),v)*dx + 0.5*dot(t*u,v)*dx
BigA = 1./dt*dot(u-uold,v)*dx   - dot(div(sigma),v)*dx + 0.5*dot(t*u,v)*dx  # + gamma*dot(u,v)*dx - dot(div(sigma),v)*dx + 0.5*dot(t*u,v)*dx

BigB = 2.*mu*inner(sym(t),s)*dx - 0.5*inner(dev(outer(u,u)),s)*dx - inner(dev(sigma),s)*dx 
BigC = inner(tau,t)*dx  + dot(u,div(tau))*dx

F  = dot(g+f,v)*dx
G  = dot(tau*nn,u_D)*ds
    

    # zero average of trace
Z  = (tr(2*sigma+outer(u,u))) * ze * dx + tr(tau) * xi * dx
    
FF = BigA + BigB + BigC  - F - G  + Z
							    
Tang = derivative(FF, Usol, Utrial)
problem = NonlinearVariationalProblem(FF, Usol, J=Tang)
solver  = NonlinearVariationalSolver(problem)
#solver.parameters['nonlinear_solver']                    = 'newton'#or snes
#solver.parameters['newton_solver']['linear_solver']      = 'mumps'
#solver.parameters['newton_solver']['absolute_tolerance'] = 1e-8
#solver.parameters['newton_solver']['relative_tolerance'] = 1e-8
#solver.parameters['newton_solver']['maximum_iterations'] = 25
prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-8
prm['newton_solver']['relative_tolerance'] = 1E-7
prm['newton_solver']['maximum_iterations'] = 25
prm['newton_solver']['relaxation_parameter'] = 1.0

# ********* Time loop ************* #


while (time <= Tfinal):
    
    print("time = %.2f" % time)
    solver.solve()
    
    uh, t11h, t12h, t13h,t21h,t22h,t23h,t31h,t32h, Rsig1h, Rsig2h,Rsig3h, xih = Usol.split()

    assign(uold,uh); 

    #if (inc % 20 == 0):
    if (inc < frequency ): 
        
        #th=as_tensor(((t11h,t12h),(t21h,-t11h)))
        th=as_tensor(((t11h,t12h,t13h),(t21h,t22h,t23h),(t31h,t32h,-t22h-t11h)))
                                
        sigmah = as_tensor((Rsig1h,Rsig2h,Rsig3h))
    
    # dimension-dependent (not separating H_0(div) with c*I)
        ph = project(-(1./6.)*tr(2*sigmah+outer(uh,uh)),Ph)

        uh.rename("u","u"); fileO.write(uh,time)
    
        ph.rename("p","p"); fileO.write(ph,time)  
        folder1 = str('/home/sebanthalas/Documents/2nd_year_2term/CFD/Archive/vtk/u_stokes'+str(inc)+'.pvd')
        vtkfile = File(folder1)
        vtkfile << uh
        folder1 = str('/home/sebanthalas/Documents/2nd_year_2term/CFD/Archive/vtk/p_stokes'+str(inc)+'.pvd')
        vtkfile = File(folder1)
        vtkfile << ph

        
    inc += 1; time += dt
    print("inc = %.2f" % inc)