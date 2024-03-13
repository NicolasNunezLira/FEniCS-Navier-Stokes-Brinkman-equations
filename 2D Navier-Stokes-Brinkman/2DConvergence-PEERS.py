'''
Convergence test for a mixed scheme of the Navier--Stokes--Brinkman equations
The domain is (0,1)x(0,1)

Manufactured smooth solutions
#######################################
strong primal form: 

 eta*u - lam*div(mu*eps(u)) + grad(u)*u + grad(p) = f  in Omega
                                           div(u) = 0  in Omega 

Pure Dirichlet conditions for u 
                                                u = u_D on Gamma

Lagrange multiplier to fix the average of p
                                           int(p) = 0

######################################

strong mixed form in terms of (t,sigma,u,gamma)

                 t + gamma = grad(u) 
lam*mu*t - dev(u otimes u) = dev(sigma)
        eta*u - div(sigma) = f
+ BC:
                         u = u_D on Gamma
+ trace of pressure:
 int(tr(sigma+u otimes u)) = 0

'''

from fenics import *
import sympy2fenics as sf
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True

# Constant coefficients 
ndim = 2
Id = Identity(ndim)

lam = Constant(1.)


# Macro operators 

epsilon = lambda v: sym(grad(v))
skewgr  = lambda v: grad(v) - epsilon(v)
str2exp = lambda s: sf.sympy2exp(sf.str2sympy(s))
curlBub = lambda vec: as_tensor([[vec[0].dx(1), -vec[0].dx(0)], [vec[1].dx(1), -vec[1].dx(0)]])

# Manufactured solutions as strings 

u_str = '(cos(pi*x)*sin(pi*y),-sin(pi*x)*cos(pi*y))'
p_str = 'x**4-y**4'#sin(x*y)'

# Initialising vectors for error history 

nkmax = 6; # max refinements
l = 0 # polynomial degree

hvec = []; nvec = []; erru = []; rateu = []; errp = []; ratep = [];
errt = []; ratet = []; errsigma = []; ratesigma = []
errgamma = []; rategamma = []; 

rateu.append(0.0); ratet.append(0.0); rategamma.append(0.0);
ratep.append(0.0); ratesigma.append(0.0);

# Error history 

for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    nps = pow(2,nk+1) 
    mesh = UnitSquareMesh(nps,nps,'crossed')
    nn   = FacetNormal(mesh)
        
    hvec.append(mesh.hmax())

    # Heterogeneous viscosity 

    mu = Expression('exp(-x[0]*x[1])', degree=3, domain = mesh)
    eta = Expression('2 + cos(x[0])*x[1]', degree = 3, domain = mesh)
    
    # Instantiation of exact solutions
    
    u_ex  = Expression(str2exp(u_str), degree=l+4, domain=mesh)
    p_ex  = Expression(str2exp(p_str), degree=l+4, domain=mesh)

    t_ex = epsilon(u_ex)
    gamma_ex = skewgr(u_ex)
    sigma_ex = lam*mu*t_ex - outer(u_ex,u_ex) - p_ex * Id

    f_ex = eta*u_ex - div(sigma_ex)
    
    # Finite element subspaces

    Ht  = VectorElement('DG', mesh.ufl_cell(), l+ndim, dim = 3)
    RTl = FiniteElement('RT', mesh.ufl_cell(), l+1)# In FEniCS, Hdiv tensors need to be defined row-wise
    Bub = VectorElement('B', mesh.ufl_cell(), l + 3) # "B_l" in (4.43)
    Hu = VectorElement('DG', mesh.ufl_cell(), l)
    Hgam = FiniteElement('CG', mesh.ufl_cell(), l+1)
    R0 = FiniteElement('R', mesh.ufl_cell(), 0)

    #product space: new part: Hsig = [RTl,RTl,Bub]
    Hh = FunctionSpace(mesh, MixedElement([Ht,RTl,RTl,Bub,Hu,Hgam,R0]))
    nvec.append(Hh.dim())
    
    # Trial and test functions (nonlinear setting)
    Trial = TrialFunction(Hh)
    Sol   = Function(Hh) 
    t_, sig1, sig2,bsol, u,gam_,xi = split(Sol)
    s_, tau1, tau2,btest, v,del_,zeta = TestFunctions(Hh)

    t = as_tensor(((t_[0], t_[1]),(t_[2],-t_[0])))
    s = as_tensor(((s_[0], s_[1]),(s_[2],-s_[0])))

    sigma = as_tensor((sig1,sig2)) + curlBub(bsol)
    tau   = as_tensor((tau1,tau2)) + curlBub(btest)

    gamma = as_tensor(((0,gam_),(-gam_,0)))
    delta = as_tensor(((0,del_),(-del_,0)))
                      
    # Essential boundary conditions: NONE FOR THIS CASE

    # Variational forms

    a   = lam*mu*inner(t,s)*dx 
    b1  = - inner(sigma,s)*dx
    b   = - inner(outer(u,u),s)*dx
    b2  = inner(t,tau)*dx
    bbt = dot(u,div(tau))*dx + inner(gamma,tau)*dx
    bb  = dot(div(sigma),v)*dx + inner(sigma,delta)*dx
    cc  = eta * dot(u,v)*dx

    AA = a + b1 + b2 + b + bbt + bb - cc + zeta*tr(sigma+outer(u,u))*dx + xi*tr(tau)*dx 
    FF = dot(tau*nn,u_ex)*ds - dot(f_ex,v)*dx + zeta*tr(sigma_ex+outer(u_ex,u_ex))*dx
    
    Nonl = AA - FF
    # Solver specifications (including essential BCs if any)

    Tangent = derivative(Nonl, Sol, Trial)
    Problem = NonlinearVariationalProblem(Nonl, Sol, J=Tangent)
    Solver  = NonlinearVariationalSolver(Problem)
    Solver.parameters['nonlinear_solver']                    = 'newton'
    Solver.parameters['newton_solver']['linear_solver']      = 'mumps'
    Solver.parameters['newton_solver']['absolute_tolerance'] = 1e-8
    Solver.parameters['newton_solver']['relative_tolerance'] = 1e-8
    Solver.parameters['newton_solver']['maximum_iterations'] = 25

    # Assembling and solving
    #solve(Nonl == 0, Sol)

    Solver.solve()
    th_, sigh1, sigh2, bsolh, uh,gamh_,xih = Sol.split()
    
    th = as_tensor(((th_[0], th_[1]),(th_[2],-th_[0])))
    sigmah = as_tensor((sigh1,sigh2)) + curlBub(bsolh)
    
    gammah = as_tensor(((0,gamh_),(-gamh_,0)))

    Ph = FunctionSpace(mesh, 'DG', l)

    # Postprocessing (eq 2.7)
    ph = project(-1/ndim*tr(sigmah + outer(uh,uh)),Ph) 

    # Error computation

    E_t = assemble(inner(t_ex - th,t_ex-th)*dx)
    E_sigma_0 = assemble(inner(sigma_ex-sigmah,sigma_ex-sigmah)*dx)
    E_sigma_div = assemble(dot(div(sigma_ex-sigmah),div(sigma_ex-sigmah))**(2./3.)*dx)
    E_u = assemble(dot(u_ex-uh,u_ex-uh)**2*dx)
    E_gamma = assemble(inner(gamma_ex-gammah,gamma_ex-gammah)*dx)
    E_p = assemble((p_ex - ph)**2*dx)

    errt.append(pow(E_t,0.5))
    errsigma.append(pow(E_sigma_0,0.5)+pow(E_sigma_div,0.75))
    erru.append(pow(E_u,0.25))
    errgamma.append(pow(E_gamma,0.5))
    errp.append(pow(E_p,0.5))

    # Computing convergence rates
    
    if(nk>0):
        ratet.append(ln(errt[nk]/errt[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        ratesigma.append(ln(errsigma[nk]/errsigma[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        rateu.append(ln(erru[nk]/erru[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        rategamma.append(ln(errgamma[nk]/errgamma[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        ratep.append(ln(errp[nk]/errp[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        
       
# Generating error history 
print('==============================================================================================================')
print('   nn  &   hh   &   e(t)   & r(t) &  e(sig)  & r(s) &   e(u)   & r(u) &  e(gam)  & r(g) &   e(p)   & r(p)  ')
print('==============================================================================================================')

for nk in range(nkmax):
    print('{:6d} & {:.4f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} '.format(nvec[nk], hvec[nk], errt[nk], ratet[nk], errsigma[nk], ratesigma[nk], erru[nk], rateu[nk], errgamma[nk], rategamma[nk], errp[nk], ratep[nk]))
print('==============================================================================================================')


'''




'''
