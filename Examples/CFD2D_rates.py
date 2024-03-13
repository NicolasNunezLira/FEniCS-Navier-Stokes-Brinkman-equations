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
import numpy as np 
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4
list_linear_solver_methods()

fileO = XDMFFile("outputs/out-NS.xdmf")
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True

# ************ Time units *********************** #

time = 0.0; Tfinal = 2000.0;  dt = 20.0; 
inc = 0;   frequency = 400;

# ****** Constant coefficients ****** #


gamma = Constant(1.0e-3)
g     = Constant((0.,-1.))
Id = Constant(((1.,0.),(0.,1.)))
# *********** Variable coefficients ********** #

mu   = Constant(1.0) #lambda phi: exp(-phi)

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

# ******* Exact solutions for error analysis ****** #
#u_str = '(4*y**2*(x**2-1)**2*(y**2-1),-4*x**2*(y**2-1)**2*(x**2-1))'
u_str = '(cos(pi/2*x)*sin(pi/2*y),-sin(pi/2*x)*cos(pi/2*y))'
p_str = '(x-0.5)*(y-0.5)-0.25'
nkmax = 4
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
for nk in range(nkmax):
    meshname="meshes/unitSqBaryRef%03g.xml"%nk
    mesh = Mesh(meshname)
    nn   = FacetNormal(mesh)
    # rates #================================================================
    hvec.append(mesh.hmax())
    #================================================================

    #================================================================
        # Exact solutions settings
    #================================================================
        
    u_ex     = Expression(str2exp(u_str), degree=7, domain=mesh)
    p_ex     = Expression(str2exp(p_str), degree=6, domain=mesh)
    t_ex     = grad(u_ex)
    sigma_ex = 2.*mu*sym(t_ex) - 0.5*outer(u_ex,u_ex) - p_ex*Id
    rhs_ex   = gamma*u_ex - div(sigma_ex) + 0.5*t_ex*u_ex -g
    #================================================================
        # Exact solutions settings
    #================================================================
    # *********** Finite Element spaces ************* #

    deg = 1
    Pkv = VectorElement('DG', mesh.ufl_cell(), deg)
    Pk  = FiniteElement('DG', mesh.ufl_cell(), deg)
    RTv = FiniteElement('RT', mesh.ufl_cell(), deg+1)
    R0  = FiniteElement('R', mesh.ufl_cell(), 0)
    Hh  = FunctionSpace(mesh, MixedElement([Pkv,Pk,Pk,Pk,RTv,RTv,R0]))
    Ph  = FunctionSpace(mesh,'CG',1)
    print (" ****** Total DoF = ", Hh.dim())
    nvec.append(Hh.dim())
    #================================================================
        # *********** Trial and test functions ********** #
    #================================================================s
    Utrial = TrialFunction(Hh)
    Usol   = Function(Hh)
    u, t11, t12, t21, Rsig1, Rsig2, xi= split(Usol)
    v, s11, s12, s21, Rtau1, Rtau2, ze= TestFunctions(Hh)

    t = as_tensor(((t11,t12),(t21,-t11)))
    s = as_tensor(((s11,s12),(s21,-s11)))
        
    sigma = as_tensor((Rsig1,Rsig2))
    tau   = as_tensor((Rtau1,Rtau2))
        # ********** Boundary conditions ******** #

        # All Dirichlet BCs become natural in this mixed form
        
        # *************** Variational forms ***************** #
            # flow equations
        
    BigA = - dot(div(sigma),v)*dx + 0.5*dot(t*u,v)*dx

    BigB = 2.*mu*inner(sym(t),s)*dx - 0.5*inner(dev(outer(u,u)),s)*dx - inner(dev(sigma),s)*dx 
    BigC = inner(tau,t)*dx  + dot(u,div(tau))*dx

    F  = dot(g+rhs_ex,v)*dx
    G  = dot(tau*nn,u_ex)*ds
        

        # zero average of trace
    Z  = (tr(2*sigma+outer(u,u))-tr(2*sigma_ex+outer(u_ex,u_ex))) * ze * dx + tr(tau) * xi * dx
        
    FF = BigA + BigB + BigC  - F - G  + Z
    							    
    Tang = derivative(FF, Usol, Utrial)

    solve(FF == 0, Usol, J=Tang)

    uh, t11h, t12h, t21h, Rsigh1, Rsigh2, xih = Usol.split()
    th=as_tensor(((t11h,t12h),(t21h,-t11h)))
    							    
    sigmah = as_tensor((Rsigh1,Rsigh2))
        
        # dimension-dependent (not separating H_0(div) with c*I)
    ph = project(-0.25*tr(2*sigmah+outer(uh,uh)),Ph)

    uh.rename("u","u"); fileO.write(uh,1.0*nk)
        
    ph.rename("p","p"); fileO.write(ph,1.0*nk)
    E_t = assemble((th-t_ex)**2*dx)
    E_p = assemble((ph-p_ex)**2*dx)
    E_s = assemble((sigma_ex-sigmah)**2*dx \
                   +(div(sigma_ex)-div(sigmah))**2*dx)
    
    erru.append(errornorm(u_ex,uh,'L2'))
    errp.append(pow(E_p,0.5))
   
    errsigma.append(pow(E_s,0.5))
    
    errt.append(pow(E_t,0.5))	
    if(nk>0):
        rateu.append(ln(erru[nk]/erru[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        ratep.append(ln(errp[nk]/errp[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        
        ratet.append(ln(errt[nk]/errt[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        
        ratesigma.append(ln(errsigma[nk]/errsigma[nk-1])/ln(hvec[nk]/hvec[nk-1]))
    folder1 = str('/home/sebanthalas/Documents/2nd_year_2term/CFD/Archive/vtk_rates/u_stokes'+str(nk)+'.pvd')
    vtkfile = File(folder1)
    vtkfile << uh
    folder1 = str('/home/sebanthalas/Documents/2nd_year_2term/CFD/Archive/vtk_rates/p_stokes'+str(nk)+'.pvd')
    vtkfile = File(folder1)
    vtkfile << ph
# ********  Generating error history **** #
print('====================================================================')
print('nn   &  hh  &   e(u)  &  r(u)  &  e(t)  &  r(t)  &  e(sig) & r(sig) ')
print('====================================================================')

for nk in range(nkmax):
    print('%d  %4.4g  %4.4g  %4.4g  %4.4g  %4.4g  %4.4g  %4.4g' % (nvec[nk], hvec[nk], erru[nk], rateu[nk], errt[nk], ratet[nk], errsigma[nk], ratesigma[nk]))
print('======================================================================')
print(' e(p) & r(p) ')
print('======================================================================')

for nk in range(nkmax):
    print('%4.4g  %4.4g' % (errp[nk], ratep[nk]))