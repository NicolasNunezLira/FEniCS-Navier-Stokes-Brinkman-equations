'''
Convergence test for a mixed scheme for the Oberbeck-Boussinesq equations
Steady
Domain is (-1,1)^2
Alfeld splits (barycentric refinements done with GMSH, see below *)
Manufactured smooth solutions

strong primal form: 

gamma u - div(2mu(phi)*eps(u)) + grad(u)*u + grad(p) = theta.phi * g
                                              div(u) = 0
                 -div(K1*grad(phi1)) + u. grad(phi1) = 0
                 -div(K2*grad(phi2)) + u. grad(phi2) = 0

Pure Dirichlet conditions for u and phi1,phi2
Lagrange multiplier to fix the average of p

strong mixed form in terms of (u,t,sigma,phi1,t1,sigma1,phi2,t2,sigma2)

grad(u) = t
gamma u - div(sigma) + 0.5*t*u - theta.phi * g = 0
2*mu(phi)*t_sym - 0.5*dev(u otimes u) = dev(sigma)
grad(phij) = tj
Kj*tj - 0.5*phij*u = sigmaj
-div(sigmaj) + 0.5*tj*u = 0
avg(tr(2*sigma+u otimes u)) = 0

RT(k) elements for the rows of sigma and for each sigmaj, and DG(k) elements for everyone else

(*) Barycentric refinement: 

gmsh generateUnitSquares.geo - 
then for each file: gmsh -barycentric_refine file.msh
then for each file: meshio-convert file.msh file.xml -z -p (remove options if in 3D)

'''

from fenics import *
import sympy2fenics as sf
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4
list_linear_solver_methods()

fileO = XDMFFile("outputs/out-Ex01Accuracy.xdmf")
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True

# ****** Constant coefficients ****** #

gamma = Constant(1.0e-3)
theta = Constant((1.,0.5))
g     = Constant((0.,-1.))
Id = Constant(((1,0),(0,1)))

# *********** Variable coefficients ********** #

mu   = lambda phi: exp(-phi) 

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

# ******* Exact solutions for error analysis ****** #
#'(4*y**2*(x**2-1)**2*(y**2-1),-4*x**2*(y**2-1)**2*(x**2-1))'
u_str = '(cos(pi/2*x)*sin(pi/2*y),-sin(pi/2*x)*cos(pi/2*y))'
p_str = '(x-0.5)*(y-0.5)-0.25'

phi1_str = 'exp(-x**2-y**2)-0.5'
phi2_str = 'exp(-x*y*(x-1)*(y-1))'

nkmax = 5; deg = 1

hvec = []; nvec = []; erru = []; rateu = []; errp = []; ratep = [];
errt = []; ratet = []; errsigma = []; ratesigma = []
errphi = []; ratephi = []; errsigmaj = []; ratesigmaj = []; 
errtj = []; ratetj = []; 

rateu.append(0.0); ratet.append(0.0); ratephi.append(0.0);
ratep.append(0.0); ratetj.append(0.0); 
ratesigma.append(0.0); ratesigmaj.append(0.0); 


# ***** Error analysis ***** #

for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    meshname="meshes/unitSqBaryRef%03g.xml"%nk
    mesh = Mesh(meshname)
    nn   = FacetNormal(mesh)
    
    hvec.append(mesh.hmax())

    K1 = Expression((("exp(-x[0])","x[0]/10."),("x[1]/10.","exp(-x[1])")),degree=2, domain=mesh)
    K2 = Expression((("exp(-x[0])","0."),("0.","exp(-x[1])")),degree=2, domain = mesh)

    # instantiation of exact solutions
    
    u_ex  = Expression(str2exp(u_str), degree=6, domain=mesh)
    p_ex  = Expression(str2exp(p_str), degree=6, domain=mesh)
    phi1_ex  = Expression(str2exp(phi1_str), degree=6, domain=mesh)
    phi2_ex  = Expression(str2exp(phi2_str), degree=6, domain=mesh)

    t_ex = grad(u_ex)
    sigma_ex = 2*mu(phi1_ex)*sym(t_ex) - 0.5*outer(u_ex,u_ex) - p_ex*Id

    phi_ex = as_vector((phi1_ex,phi2_ex))
    t1_ex = grad(phi1_ex)
    t2_ex = grad(phi2_ex)
    
    rhs_ex = gamma*u_ex - div(sigma_ex) + 0.5*t_ex*u_ex - dot(theta,phi_ex)*g
    
    sigma1_ex = K1*t1_ex - 0.5*phi1_ex*u_ex
    sigma2_ex = K2*t2_ex - 0.5*phi2_ex*u_ex
    rhs1_ex = -div(sigma1_ex) + 0.5*dot(t1_ex,u_ex)
    rhs2_ex = -div(sigma2_ex) + 0.5*dot(t2_ex,u_ex)

    # *********** Finite Element spaces ************* #
    # because of current fenics syntax, we need to define the rows
    # of sigma separately
    
    Pkv = VectorElement('DG', mesh.ufl_cell(), deg)
    Pk = FiniteElement('DG', mesh.ufl_cell(), deg)
    Rtv = FiniteElement('RT', mesh.ufl_cell(), deg+1)
    R0 = FiniteElement('R', mesh.ufl_cell(), 0)

    Hh = FunctionSpace(mesh, MixedElement([Pkv,Pk,Pk,Pk,Rtv,Rtv,Pk,Pkv,Rtv,Pk,Pkv,Rtv,R0]))
    Ph = FunctionSpace(mesh,'CG',1)
    print (" ****** Total DoF = ", Hh.dim())

    nvec.append(Hh.dim())
    
    # *********** Trial and test functions ********** #

    Utrial = TrialFunction(Hh)
    Usol = Function(Hh)
    u, t11, t12, t21, Rsig1, Rsig2, phi1, t1, sigma1, phi2, t2, sigma2, xi = split(Usol)
    v, s11, s12, s21, Rtau1, Rtau2, psi1, s1,   tau1, psi2, s2,   tau2, ze = TestFunctions(Hh)

    t = as_tensor(((t11,t12),(t21,-t11)))
    s = as_tensor(((s11,s12),(s21,-s11)))
    phi = as_vector((phi1,phi2))
    sigma = as_tensor((Rsig1,Rsig2))
    tau   = as_tensor((Rtau1,Rtau2))
    
    # ********** Boundary conditions ******** #

    # All Dirichlet BCs become natural in this mixed form
    
    # *************** Variational forms ***************** #

    # flow equations
    
    Aphi = gamma*dot(u,v)*dx + 2*mu(phi1)*inner(sym(t),s)*dx
    C  = 0.5*dot(t*u,v)*dx - 0.5*inner(dev(outer(u,u)),dev(s))*dx
    Bt = - inner(dev(sigma),s)*dx - dot(div(sigma),v)*dx
    B  = - inner(tau,t)*dx  - dot(u,div(tau))*dx

    F  = dot(theta,phi)*dot(g,v)*dx + dot(rhs_ex,v)*dx
    G  = - dot(tau*nn,u_ex)*ds
    
    # temperature
    Aj1 = dot(K1*t1,s1)*dx
    Cu1 = 0.5*psi1*dot(t1,u)*dx - 0.5*phi1*dot(u,s1)*dx
    B1t = - dot(sigma1,s1)*dx - psi1*div(sigma1)*dx
    B1  = - dot(tau1,t1)*dx - phi1*div(tau1)*dx

    F1 = rhs1_ex*psi1*dx
    G1 = - dot(tau1,nn)*phi1_ex*ds

    # concentration
    Aj2 = dot(K2*t2,s2)*dx
    Cu2 = 0.5*psi2*dot(t2,u)*dx - 0.5*phi2*dot(u,s2)*dx
    B2t = - dot(sigma2,s2)*dx - psi2*div(sigma2)*dx
    B2  = - dot(tau2,t2)*dx - phi2*div(tau2)*dx
    
    F2 = rhs2_ex*psi2*dx
    G2 = - dot(tau2,nn)*phi2_ex*ds

    # zero (or, in this case, exact) average of trace
    Z  = (tr(2*sigma+outer(u,u))-tr(2*sigma_ex+outer(u_ex,u_ex)))* ze * dx + tr(tau) * xi * dx
    
    FF = Aphi + C + Bt + B \
         - F - G \
         + Aj1 + Cu1 + B1t + B1 \
         - F1 - G1 \
         + Aj2 + Cu2 + B2t + B2 \
         - F2 - G2 \
         + Z
    
    Tang = derivative(FF, Usol, Utrial)
    problem = NonlinearVariationalProblem(FF, Usol, J=Tang) #no BCs
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters['nonlinear_solver']                    = 'newton'#or snes
    solver.parameters['newton_solver']['linear_solver']      = 'mumps'
    solver.parameters['newton_solver']['absolute_tolerance'] = 1e-8
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-8
    solver.parameters['newton_solver']['maximum_iterations'] = 25
    
    solver.solve()

    uh, t11h, t12h, t21h, Rsigh1, Rsigh2, phi1h, t1h, sigma1h, phi2h, t2h, sigma2h, xih = Usol.split()
    th=as_tensor(((t11h,t12h),(t21h,-t11h)))
    phih = as_vector((phi1h,phi2h))
    sigmah = as_tensor((Rsigh1,Rsigh2))
    
    # dimension-dependent (not separating H_0(div) with c*I)
    ph = project(-0.25*tr(2*sigmah+outer(uh,uh)),Ph)

    uh.rename("u","u"); fileO.write(uh,1.0*nk)
    phi1h.rename("phi1","phi1"); fileO.write(phi1h,1.0*nk)
    phi2h.rename("phi2","phi2"); fileO.write(phi2h,1.0*nk)
    ph.rename("p","p"); fileO.write(ph,1.0*nk)
    
    E_t = assemble((th-t_ex)**2*dx)
    E_phi = assemble((phih-phi_ex)**2*dx)
    E_tj = assemble((t1h-t1_ex)**2*dx + (t2h-t2_ex)**2*dx)
    E_sj = assemble((sigma1_ex-sigma1h)**2*dx \
                    + (sigma2_ex-sigma2h)**2*dx \
                    + (div(sigma1_ex)-div(sigma1h))**2*dx \
                    + (div(sigma2_ex)-div(sigma2h))**2*dx)
    
    E_p = assemble((ph-p_ex)**2*dx)
    E_s = assemble((sigma_ex-sigmah)**2*dx \
                   +(div(sigma_ex)-div(sigmah))**2*dx)
    
    erru.append(errornorm(u_ex,uh,'L2'))
    errp.append(pow(E_p,0.5))
    errphi.append(pow(E_phi,0.5))
    errsigma.append(pow(E_s,0.5))
    errsigmaj.append(pow(E_sj,0.5))
    errt.append(pow(E_t,0.5))
    errtj.append(pow(E_tj,0.5))
    
    if(nk>0):
        rateu.append(ln(erru[nk]/erru[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        ratep.append(ln(errp[nk]/errp[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        ratephi.append(ln(errphi[nk]/errphi[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        ratet.append(ln(errt[nk]/errt[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        ratetj.append(ln(errtj[nk]/errtj[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        ratesigma.append(ln(errsigma[nk]/errsigma[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        ratesigmaj.append(ln(errsigmaj[nk]/errsigmaj[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        
# ********  Generating error history **** #
print('====================================================================')
print('nn   &  hh  &   e(u)  &  r(u)  &  e(t)  &  r(t)  &  e(sig) & r(sig) ')
print('====================================================================')

for nk in range(nkmax):
    print('%d  %4.4g  %4.4g  %4.4g  %4.4g  %4.4g  %4.4g  %4.4g' % (nvec[nk], hvec[nk], erru[nk], rateu[nk], errt[nk], ratet[nk], errsigma[nk], ratesigma[nk]))
print('======================================================================')
print('e(phi)  & r(phi)  & e(tj)  & r(tj)  & e(sigj) & r(sigj) & e(p) & r(p) ')
print('======================================================================')

for nk in range(nkmax):
    print('%4.4g  %4.4g  %4.4g  %4.4g  %4.4g  %4.4g %4.4g  %4.4g' % (errphi[nk], ratephi[nk], errtj[nk], ratetj[nk], errsigmaj[nk], ratesigmaj[nk], errp[nk], ratep[nk]))
    

'''
====================================================================
nn   &  hh  &   e(u)  &  r(u)  &  e(t)  &  r(t)  &  e(sig) & r(sig) 
====================================================================
1305  1.414  0.09025     0  0.556     0  0.9379     0
5153  0.7071  0.01935  2.222  0.199  1.482  0.2785  1.752
20481  0.3536  0.004202  2.203  0.0622  1.678  0.07634  1.867
81665  0.1768  0.0009708  2.114  0.01849  1.75  0.02052  1.895
326145  0.08839  0.0002337  2.054  0.00513  1.849  0.005364  1.936
1303553  0.04419  0.0001405  2.003  0.00137  1.94  0.001649  1.979
======================================================================
e(phi)  & r(phi)  & e(tj)  & r(tj)  & e(sigj) & r(sigj) & e(p) & r(p) 
======================================================================
0.0769     0  0.5343     0  0.7424     0 0.1795     0
0.01586  2.278  0.1681  1.668  0.2094  1.826 0.04699  1.934
0.00334  2.247  0.04739  1.827  0.05768  1.86 0.009763  2.267
0.0007648  2.126  0.01271  1.898  0.01519  1.925 0.002147  2.185
0.000185  2.048  0.003318  1.938  0.003891  1.965 0.0005073  2.081
0.0001064  2.002  0.0008971  1.966  0.00143  1.993 0.0001324  2.03

'''

