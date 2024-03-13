'''
Convergence test for a mixed scheme for the Oberbeck-Boussinesq equations
Steady 3D
Domain is (0,1)x(0,0.5)x(0,0.5)
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

RT(k) elements for the rows of sigma and for each sigmaj, and DG(k) elements for everyone else (k>2 in 3D)

(*) Barycentric refinement: 

gmsh generateUnitSquares.geo - 
then for each file: gmsh -barycentric_refine file.msh
then for each file: meshio-convert file.msh file.xml (no need for flatten options in 3D)

'''

from fenics import *
import sympy2fenics as sf
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4
list_linear_solver_methods()

fileO = XDMFFile("outputs/out-Ex03Convergence.xdmf")
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True

# ****** Constant coefficients ****** #

gamma = Constant(1.0)
theta = Constant((1.,0.5))
g     = Constant((0.,0.,-1.))
Id = Constant(((1,0,0),(0,1,1),(0,0,1)))

# *********** Variable coefficients ********** #

mu   = lambda phi1,phi2: 0.1+exp(-phi1+phi2) 

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

# ******* Exact solutions for error analysis ****** #
u_str = '(sin(pi*x)*cos(pi*y)*cos(pi*z), -2*cos(pi*x)*sin(pi*y)*cos(pi*z), cos(pi*x)*cos(pi*y)*sin(pi*z))'
p_str = 'sin(pi*x)*sin(pi*y)*sin(pi*z) - 0.0645'
phi1_str = '1-sin(pi*x)*cos(pi*y)*sin(pi*z)'
phi2_str = 'exp(-(x-0.5)**2-(y-0.25)**2-(z-0.25)**2)'

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
    meshname="meshes/boxRef%03g.xml"%nk
    mesh = Mesh(meshname)
    nn   = FacetNormal(mesh)
    
    hvec.append(mesh.hmax())

    K1 = Expression((("exp(-x[0])","0.","0"),\
                     ("0.","exp(-x[1])","0"),\
                     ("0.","0.", "exp(-x[2])")), \
                    degree=7, domain = mesh)
    K2 = Constant(1.0)

    # instantiation of exact solutions
    
    u_ex  = Expression(str2exp(u_str), degree=8, domain=mesh)
    p_ex  = Expression(str2exp(p_str), degree=8, domain=mesh)

    # |Omega| = 0.25, n = 3

    ccx = assemble(-1./(6.*0.25)*dot(u_ex,u_ex)*1.0*dx)
    
    phi1_ex  = Expression(str2exp(phi1_str), degree=8, domain=mesh)
    phi2_ex  = Expression(str2exp(phi2_str), degree=8, domain=mesh)

    t_ex = grad(u_ex)
    sigma_ex = 2*mu(phi1_ex,phi2_ex)*sym(t_ex) - 0.5*outer(u_ex,u_ex) - p_ex*Id \
               - ccx*Id

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

    Hh = FunctionSpace(mesh, MixedElement([Pkv,Pk,Pk,Pk,Pk,Pk,Pk,Pk,Pk,Rtv,Rtv,Rtv,Pk,Pkv,Rtv,Pk,Pkv,Rtv,R0]))

    # spaces to project for visualisation only
    Ph = FunctionSpace(mesh,'CG',1)
    TTh = TensorFunctionSpace(mesh,'CG',1)
    VVh = VectorFunctionSpace(mesh,'CG',1)
    
    print (" ****** Total DoF = ", Hh.dim())

    nvec.append(Hh.dim())
    
    # *********** Trial and test functions ********** #

    Utrial = TrialFunction(Hh)
    Usol = Function(Hh)
    u, t11, t12, t13, t21, t22, t23, t31, t32, Rsig1, Rsig2, Rsig3, phi1, t1, sigma1, phi2, t2, sigma2, xi = split(Usol)
    v, s11, s12, s13, s21, s22, s23, s31, s32, Rtau1, Rtau2, Rtau3, psi1, s1,   tau1, psi2, s2,   tau2, ze = TestFunctions(Hh)

    t=as_tensor(((t11,t12,t13),(t21,t22,t23),(t31,t32,-t11-t22)))
    s=as_tensor(((s11,s12,s13),(s21,s22,s23),(s31,s32,-s11-s22)))
    
    phi = as_vector((phi1,phi2))
    sigma = as_tensor((Rsig1,Rsig2,Rsig3))
    tau   = as_tensor((Rtau1,Rtau2,Rtau3))
    
    # ********** Boundary conditions ******** #

    # All Dirichlet BCs become natural in this mixed form
    
    # *************** Variational forms ***************** #

    # flow equations
    
    Aphi = gamma*dot(u,v)*dx + 2*mu(phi1,phi2)*inner(sym(t),s)*dx
    C  = 0.5*dot(t*u,v)*dx - 0.5*inner(dev(outer(u,u)),dev(s))*dx
    Bt = - inner(sigma,s)*dx - dot(div(sigma),v)*dx
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

    uh, t11h, t12h, t13h, t21h, t22h, t23h, t31h, t32h, Rsigh1, Rsigh2, Rsigh3, phi1h, t1h, sigma1h, phi2h, t2h, sigma2h, xih = Usol.split()

    th=as_tensor(((t11h,t12h,t13h),(t21h,t22h,t23h),(t31h,t32h,-t11h-t22h)))
    phih = as_vector((phi1h,phi2h))
    sigmah = as_tensor((Rsigh1,Rsigh2,Rsigh3))
    
    cch = assemble(-1./(6.*0.25)*dot(uh,uh)*dx)
    
    # dimension-dependent
    ph = project(-1./6.*tr(2.*sigmah+2.*cch*Id+outer(uh,uh)),Ph)

    # projecting for visualisation
    th_out=project(th,TTh)
    sh_out=project(sigmah,TTh)
    t1h_out= project(t1h,VVh)
    t2h_out= project(t2h,VVh)
    s1h_out=project(sigma1h,VVh)
    s2h_out=project(sigma2h,VVh)
    
    uh.rename("u","u"); fileO.write(uh,1.0*nk)
    phi1h.rename("phi1","phi1"); fileO.write(phi1h,1.0*nk)
    phi2h.rename("phi2","phi2"); fileO.write(phi2h,1.0*nk)
    ph.rename("p","p"); fileO.write(ph,1.0*nk)
    th_out.rename("t","t"); fileO.write(th_out,1.0*nk)
    sh_out.rename("sig","sig"); fileO.write(sh_out,1.0*nk)
    t1h_out.rename("t1","t1"); fileO.write(t1h_out,1.0*nk)
    t2h_out.rename("t2","t2"); fileO.write(t2h_out,1.0*nk)
    s1h_out.rename("sig1","sig1"); fileO.write(s1h_out,1.0*nk)
    s2h_out.rename("sig2","sig2"); fileO.write(s2h_out,1.0*nk)

    E_u = assemble((uh-u_ex)**2*dx)
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

    erru.append(pow(E_u,0.5)) 
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
    print('%d & %4.4g & %4.4g & %4.4g & %4.4g & %4.4g & %4.4g & %4.4g' % (nvec[nk], hvec[nk], erru[nk], rateu[nk], errt[nk], ratet[nk], errsigma[nk], ratesigma[nk]))
print('======================================================================')
print('e(phi)  & r(phi)  & e(tj)  & r(tj)  & e(sigj) & r(sigj) & e(p) & r(p) ')
print('======================================================================')

for nk in range(nkmax):
    print('%4.4g & %4.4g & %4.4g & %4.4g & %4.4g & %4.4g & %4.4g & %4.4g' % (errphi[nk], ratephi[nk], errtj[nk], ratetj[nk], errsigmaj[nk], ratesigmaj[nk], errp[nk], ratep[nk]))

'''

for k=1:
====================================================================
nn   &  hh  &   e(u)  &  r(u)  &  e(t)  &  r(t)  &  e(sig) & r(sig) 
====================================================================
2995 & 1.225 & 0.0945 &    0 & 1.417 &    0 & 6.47 &    0
5959 & 0.866 & 0.05992 & 1.314 & 0.577 & 2.592 & 4.212 & 1.239
47065 & 0.433 & 0.01602 & 1.904 & 0.2295 & 1.33 & 1.212 & 1.797
374113 & 0.2165 & 0.004083 & 1.972 & 0.09109 & 1.333 & 0.3652 & 1.731
2983297 & 0.1083 & 0.00114 & 1.841 & 0.03185 & 1.516 & 0.097 & 1.8908
======================================================================
e(phi)  & r(phi)  & e(tj)  & r(tj)  & e(sigj) & r(sigj) & e(p) & r(p) 
======================================================================
0.05534 &    0 & 0.2565 &    0 & 1.102 &    0 & 0.3012 &    0
0.02486 & 2.309 & 0.1497 & 1.555 & 0.5431 & 2.041 & 0.1084 & 1.526
0.006622 & 1.909 & 0.0418 & 1.84 & 0.1467 & 1.889 & 0.0275 & 1.664
0.001678 & 1.981 & 0.01082 & 1.949 & 0.03745 & 1.97 & 0.0076 & 1.551
0.0004208 & 1.995 & 0.002741 & 1.981 & 0.00942 & 1.991 & 0.00212 & 1.557



k = 2: 
====================================================================
nn   &  hh  &   e(u)  &  r(u)  &  e(t)  &  r(t)  &  e(sig) & r(sig) 
====================================================================
7621 & 1.225 & 0.03165 &    0 & 0.5557 &    0 & 2.453 &    0
15181 & 0.866 & 0.01095 & 3.063 & 0.1847 & 3.178 & 0.8475 & 3.066
120241 & 0.433 & 0.001432 & 2.935 & 0.0378 & 2.289 & 0.1554 & 2.73
957121 & 0.2165 & 0.0005659 & 1.339 & 0.01084 & 1.801 & 0.0288 & 2.833
======================================================================
e(phi)  & r(phi)  & e(tj)  & r(tj)  & e(sigj) & r(sigj) & e(p) & r(p) 
======================================================================
0.01043 &    0 & 0.1079 &    0 & 0.221 &    0 & 0.2351 &    0
0.003997 & 2.766 & 0.0338 & 3.35 & 0.06753 & 3.421 & 0.03219 & 2.895
0.0004898 & 3.029 & 0.005183 & 2.705 & 0.009302 & 2.86 & 0.00464 & 2.7074
5.99e-05 & 3.032 & 0.0007968 & 2.701 & 0.001302 & 2.837 & 0.000932 & 2.848



'''
