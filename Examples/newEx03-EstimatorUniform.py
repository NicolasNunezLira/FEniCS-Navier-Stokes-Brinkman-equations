'''
Convergence test for a mixed scheme for the Oberbeck-Boussinesq equations
Steady
Domain is (-1,1)^2
Alfeld splits (barycentric refinements done with GMSH, see below *)
Manufactured smooth solutions

Computing A-posteriori error estimator, but refining uniformly

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

(*) Barycentric refinement, uniform case 

gmsh generateLShaped.geo - 

then for each file: gmsh -barycentric_refine file.msh
then for each file: meshio-convert file.msh file.xml -z -p (remove options if in 3D)

'''

from fenics import *
import sympy2fenics as sf
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4
list_linear_solver_methods()

fileO = XDMFFile("outputs/out-Ex01B-AccuracyLShaped.xdmf")
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True

# ****** Constant coefficients ****** #

gamma = Constant(1.0e-3)
theta = Constant((1.,0.5))
g     = Constant((0.,-1.))
Id = Constant(((1,0),(0,1)))

# *********** Variable coefficients ********** #

mu   = lambda phi: exp(-phi)

Tcurl    = lambda ten: as_vector((Dx(ten[0,1],0)-Dx(ten[0,0],1),Dx(ten[1,1],0)-Dx(ten[1,0],1)))

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

# ******* Exact solutions for error analysis ****** #

u_str = '(cos(pi*x/2)*sin(pi*y/2),-sin(pi*x/2)*cos(pi*y/2))'
p_str = '(2+sin(x*y))/((x-0.01)**2+(y-0.01)**2)'

phi1_str = 'exp(-x**2-y**2)-cos(x*y)'
phi2_str = 'exp(-100*(x-0.01)**2-100*(y-0.01)**2)'

nkmax = 5; deg = 1

hvec = []; nvec = []; erru = []; rateu = []; errp = []; ratep = [];
errt = []; ratet = []; errsigma = []; ratesigma = []
errphi = []; ratephi = []; errsigmaj = []; ratesigmaj = []; 
errtj = []; ratetj = []; indicator = []; etot = []; 

rateu.append(0.0); ratet.append(0.0); ratephi.append(0.0);
ratep.append(0.0); ratetj.append(0.0); 
ratesigma.append(0.0); ratesigmaj.append(0.0); 


# ***** Error analysis ***** #

for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    meshname="meshes/LshapedBaryRef%03g.xml"%nk
    mesh = Mesh(meshname)
    nn   = FacetNormal(mesh)
    tan  = as_vector((-nn[1],nn[0]))
    
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
    solver.parameters['newton_solver']['linear_solver']      = 'lu'
    solver.parameters['newton_solver']['absolute_tolerance'] = 1e-7
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-7
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

    etot.append(pow(pow(errornorm(u_ex,uh,'L2'),2)+E_t+E_s+E_phi+E_sj+E_tj,0.5))

    # ********* Error estimation  ******** #
    
    Adh = FunctionSpace(mesh, "DG", 0)
    wh = TestFunction(Adh)
    hK = CellDiameter(mesh)
    he = FacetArea(mesh)

    PsiA = wh * (sqrt((gamma*uh-div(sigmah)+0.5*th*uh \
                       - dot(theta,phih)*g-rhs_ex)**2))**(4./3.)*dx \
           + wh * abs(0.5*dot(t1h,uh) - div(sigma1h)-rhs1_ex)**(4./3.)*dx \
           + wh * abs(0.5*dot(t2h,uh) - div(sigma2h)-rhs2_ex)**(4./3.)*dx
    
    PsiB = wh * (2*mu(phi1h)*sym(th) - 0.5*dev(outer(uh,uh))-dev(sigmah))**2*dx \
           + wh * hK**2 * Tcurl(th)**2*dx \
           + avg(wh) * avg(he) * jump(th,tan)**2*dS \
           + wh * he * (th*tan - grad(u_ex)*tan)**2*ds \
           + wh * (K1*t1h-0.5*phi1h*uh-sigma1h)**2*dx \
           + wh * (K2*t2h-0.5*phi2h*uh-sigma2h)**2*dx \
           + wh * hK**2*rot(t1h)**2*dx \
           + wh * hK**2*rot(t2h)**2*dx \
           + avg(wh) * avg(he) * jump(t1h,tan)**2 * dS \
           + avg(wh) * avg(he) * jump(t2h,tan)**2 * dS \
           + wh * he * dot(t1h - grad(phi1_ex),tan)**2 * ds \
           + wh * he * dot(t2h - grad(phi2_ex),tan)**2 * ds 
           
    PsiC = wh * hK**4 * ((th - grad(uh))**2)**2*dx \
           + wh * he**2 * ((uh - u_ex)**2)**2 * ds \
           + wh * hK**4 * ((t1h - grad(phi1h))**2)**2 * dx\
           + wh * hK**4 * ((t2h - grad(phi2h))**2)**2 * dx\
           + wh * he**2 * (phi1h - phi1_ex)**4 * ds \
           + wh * he**2 * (phi2h - phi2_ex)**4 * ds

    glPsiA = assemble(PsiA);
    glPsiB = assemble(PsiB); glPsiC = assemble(PsiC)
    
    Psi = pow(sum(glPsiA),0.75)+pow(sum(glPsiB),0.5)+pow(sum(glPsiC),0.25)

    print("error_estimate = ", Psi)
    indicator.append(Psi)
    
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
print('e(phi)  & r(phi)  & e(tj)  & r(tj)  & e(sigj) & r(sigj) & e(p) & r(p) & eff ')
print('======================================================================')

for nk in range(nkmax):
    print('%4.4g  %4.4g  %4.4g  %4.4g  %4.4g  %4.4g %4.4g  %4.4g %4.4g' % (errphi[nk], ratephi[nk], errtj[nk], ratetj[nk], errsigmaj[nk], ratesigmaj[nk], errp[nk], ratep[nk], etot[nk]/indicator[nk]))
    

'''
With smooth functions

u_str = '(cos(pi/2*x)*sin(pi/2*y),-sin(pi/2*x)*cos(pi/2*y))'
p_str = '(x-0.5)*(y-0.5)-0.25'

phi1_str = 'exp(-x**2-y**2)-0.5'
phi2_str = 'exp(-x*y*(x-1)*(y-1))'

 and with uniform refinement

====================================================================
nn   &  hh  &   e(u)  &  r(u)  &  e(t)  &  r(t)  &  e(sig) & r(sig) 
====================================================================
1941     1  0.05566     0  0.1793     0  0.4819     0
7697   0.5  0.01399  1.993  0.0551  1.702  0.1161  2.053
30657  0.25  0.003498     2  0.01859  1.568  0.03245  1.839
122369  0.125  0.000873  2.002  0.006354  1.548  0.009098  1.835
488961  0.0625  0.0002178  2.003  0.001948  1.705  0.002475  1.878
======================================================================
e(phi)  & r(phi)  & e(tj)  & r(tj)  & e(sigj) & r(sigj) & e(p) & r(p) & eff 
======================================================================
0.08485     0  0.4976     0  1.033     0 0.07584     0 0.2694
0.01564  2.44  0.1388  1.842  0.2212  2.223 0.01371  2.468 0.2194
0.003686  2.085  0.04181  1.731  0.06253  1.823 0.003917  1.807 0.2187
0.0008841  2.06  0.01115  1.907  0.01581  1.983 0.00103  1.928 0.2215
0.0002178  2.021  0.002854  1.966  0.003968  1.995 0.0002508  2.037 0.2238


with 

u_str = '(cos(pi*x)*sin(pi*y),-sin(pi*x)*cos(pi*y))'
p_str = '(2+sin(x*y))/((x-0.02)**2+(y-0.02)**2)'

phi1_str = 'exp(-x**2-y**2)-cos(y)'
phi2_str = 'exp(-100*(x-0.02)**2-100*(y-0.02)**2)'

====================================================================
nn   &  hh  &   e(u)  &  r(u)  &  e(t)  &  r(t)  &  e(sig) & r(sig) 
====================================================================
1941     1  3.806     0  41.98     0  1055     0
7697   0.5  4.986  -0.3896  76.84  -0.8721  1546  -0.5515
30657  0.25  2.889  0.7873  81.93  -0.09238  3054  -0.9825
122369  0.125  0.4919  2.554  40.35  1.022  4474  -0.5506
488961  0.0625  0.1384  1.829  22.04  0.8726  2182  1.036
======================================================================
e(phi)  & r(phi)  & e(tj)  & r(tj)  & e(sigj) & r(sigj) & e(p) & r(p) & eff 
======================================================================
0.5446     0  2.957     0  31.35     0 124.9     0 0.9074
0.1063  2.356  1.488  0.991  22.47  0.4801 70.68  0.8214 1.152
0.05036  1.078  0.5917  1.33  13.02  0.788 79.91  -0.1769 1.781
0.008208  2.617  0.1788  1.727  2.509  2.375 59.19  0.4331 3.141
0.001549  2.406  0.05114  1.806  0.7944  1.659 30.39  0.9618 4.142

'''

