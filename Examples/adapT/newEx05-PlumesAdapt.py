'''
Double-diffusive flows as in Lenarda, Paggi, RB (JCP 2017)

strong primal form: 

gamma u - div(2mu(phi)*eps(u)) + grad(u)*u + grad(p) = theta.phi * g
                                              div(u) = 0
                 -div(K1*grad(phi1)) + u. grad(phi1) = 0
                 -div(K2*grad(phi2)) + u. grad(phi2) = 0

BCs?

strong mixed form in terms of (u,t,sigma,phi1,t1,sigma1,phi2,t2,sigma2)

grad(u) = t
partial_t u + gamma u - div(sigma) + 0.5*t*u - theta.phi * g = 0
2*mu*t_sym - 0.5*dev(u otimes u) = dev(sigma)
grad(phij) = tj
Kj*tj - 0.5*phij*u = sigmaj
partial_t phij-div(sigmaj) + 0.5*tj*u = 0
avg(tr(2*sigma+u otimes u)) = 0

RT(k) elements for the rows of sigma and for each sigmaj, and DG(k) elements for everyone else

Adaptive mesh refinement

CAREFUL!!!! : 
phi1 is now concentration and phi2 temperature.
Sorry for the confusion (needed to keep the structure from Lenarda JCP17)

'''

from fenics import *
import numpy as np
import os
parameters["form_compiler"]["representation"] = "uflacs"
parameters["refinement_algorithm"] = "plaza_with_parent_facets" # for adaptive!
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2

fileO = XDMFFile("outputs/out-Ex05PlumesAdapt.xdmf")
fileO.parameters["functions_share_mesh"] = True
fileO.parameters['rewrite_function_mesh'] = True # for adaptive!
fileO.parameters["flush_output"] = True

# ************ Time units *********************** #

time = 0.0; Tfinal = 2000.0;  dt = 20.0; 
inc = 0;   frequency = 20;

# ****** Constant coefficients ****** #


mu = Constant(1.)
Le = Constant(8.)
Da = Constant(0.001)
gammaT = Constant(5.)
g  = Constant((0.,-1.))

cero = Constant((0.,0.))

K2 = Le
K1 = Constant(2.5)

# ********** Mesh loading **************** #

#Omega = (0,2000)x(-1000,0)
mesh = RectangleMesh(Point(0,-1000),Point(2000,0),32,16)
nn = FacetNormal(mesh)

# *********** Variable coefficients ********** #

f    = lambda phi: phi*(1.+7.*phi)*(1.-phi)**2

# *********** Finite Element spaces ************* #
# because of current fenics syntax, we need to define the rows
# of sigma separately

deg = 0

Pkv = VectorElement('DG', mesh.ufl_cell(), deg)
Pk = FiniteElement('DG', mesh.ufl_cell(), deg)
Rtv = FiniteElement('RT', mesh.ufl_cell(), deg+1)
R0 = FiniteElement('R', mesh.ufl_cell(), 0)

Hh = FunctionSpace(mesh, MixedElement([Pkv,Pk,Pk,Pk,Rtv,Rtv,Pk,Pkv,Rtv,Pk,Pkv,Rtv,R0]))
Ph = FunctionSpace(mesh,'CG',1)
Rh = FunctionSpace(mesh,'DG',deg)

print (" ****** Total DoF = ", Hh.dim())
    
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

# ********** Initial conditions ******** #

uold = Function(Hh.sub(0).collapse())
phi1old = Function(Rh)
phi2old = Function(Rh)
perth = Function(Rh)
pert = np.random.uniform(-1,1,phi1old.vector().get_local().size)
perth.vector()[:] = 0.999+pert*0.001
init = Expression("(x[1]>-80 && x[1]<-40) ? i1 : 0.", i1=perth, degree=2)

phi1old = interpolate(init,Rh)
phi2old = interpolate(init,Rh)

gamma = Constant(1.)#interpolate(1./(1.+0.25*(perth/0.001-0.999)),Rh)

# ******* Boundaries and boundary conditions ******** #

bdry = MeshFunction("size_t", mesh, 1)
bdry.set_all(0)

bot = 31; top =32; wall= 33

GTop = CompiledSubDomain("near(x[1],0.) && on_boundary")
GBot = CompiledSubDomain("near(x[1],-1000.) && on_boundary")
GWall = CompiledSubDomain("(near(x[0],0.) || near(x[0],2000.)) && on_boundary")
GTop.mark(bdry,top); GBot.mark(bdry,bot); GWall.mark(bdry,wall)

ds = Measure("ds", subdomain_data = bdry)

# NO-FLUX BCs can be either for the total fluxes sigmaj or for the
# diffusive fluxes tj

bs1a    = DirichletBC(Hh.sub(8), cero, bdry, wall)
bs2a    = DirichletBC(Hh.sub(11), cero, bdry, wall)

bcD = [bs1a,bs2a]

u_D = cero
phi1_bot = Constant(0.)
phi2_bot = Constant(0.)
phi1_top = Constant(1.)
phi2_top = Constant(1.)
    
# *************** Variational forms ***************** #

# flow equations
    
Aphi = 1./dt*dot(u-uold,v)*dx \
       + gamma * dot(u,v) * dx + 2*mu*inner(sym(t),s)*dx
C  = 0.5*dot(t*u,v)*dx - 0.5*inner(dev(outer(u,u)),dev(s))*dx
Bt = - inner(dev(sigma),s)*dx - dot(div(sigma),v)*dx
B  = - inner(tau,t)*dx  - dot(u,div(tau))*dx

F  = (gammaT*phi2-phi1)*dot(g,v)*dx 
G  = - dot(tau*nn,u_D)*ds
    
# concentration
Aj1 = 1./dt*(phi1-phi1old)*psi1*dx \
      + dot(K1*t1,s1)*dx
Cu1 = 0.5*psi1*dot(t1,u)*dx - 0.5*phi1*dot(u,s1)*dx
B1t = - dot(sigma1,s1)*dx - psi1*div(sigma1)*dx
B1  = - dot(tau1,t1)*dx - phi1*div(tau1)*dx

F1 = -Da*f(phi1)*psi1*dx #reaction
G1 = - dot(tau1,nn)*phi1_bot*ds(bot) - dot(tau1,nn)*phi1_top*ds(top)

# temperature
Aj2 = 1./dt*(phi2-phi2old)*psi2*dx \
      + dot(K2*t2,s2)*dx
Cu2 = 0.5*psi2*dot(t2,u)*dx - 0.5*phi2*dot(u,s2)*dx
B2t = - dot(sigma2,s2)*dx - psi2*div(sigma2)*dx
B2  = - dot(tau2,t2)*dx - phi2*div(tau2)*dx
    
F2 = Da*f(phi1)*psi2*dx #reaction
G2 = - dot(tau2,nn)*phi2_bot*ds(bot) - dot(tau2,nn)*phi2_top*ds(top)

# zero average of trace
Z  = tr(2*sigma+outer(u,u)) * ze * dx + tr(tau) * xi * dx
    
FF = Aphi + C + Bt + B \
     - F - G \
     + Aj1 + Cu1 + B1t + B1 \
     - F1 - G1 \
     + Aj2 + Cu2 + B2t + B2 \
     - F2 - G2 \
     + Z
    
Tang = derivative(FF, Usol, Utrial)
problem = NonlinearVariationalProblem(FF, Usol, bcD, J=Tang)
solver  = NonlinearVariationalSolver(problem)
solver.parameters['nonlinear_solver']                    = 'newton'#or snes
solver.parameters['newton_solver']['linear_solver']      = 'petsc'
solver.parameters['newton_solver']['absolute_tolerance'] = 1e-8
solver.parameters['newton_solver']['relative_tolerance'] = 1e-8
solver.parameters['newton_solver']['maximum_iterations'] = 25


adaptSteps = 5
tolAdapt = 1.0E-6
ref_ratio = 0.1

# ********* Time loop ************* #

while (time <= Tfinal):
    
    print("time = %.2f" % time)
    solver.solve()
    
    uh, t11h, t12h, t21h, Rsigh1, Rsigh2, phi1h, t1h, sigma1h, phi2h, t2h, sigma2h, xih = Usol.split()

    assign(uold,uh); assign(phi1old,phi1h); assign(phi2old,phi2h); 

    if (inc % frequency == 0):
        
        #th=as_tensor(((t11h,t12h),(t21h,-t11h)))
        sigmah = as_tensor((Rsigh1,Rsigh2))
        ph = project(-0.25*tr(2*sigmah+outer(uh,uh)),Ph)

        uh.rename("u","u"); fileO.write(uh,time)
        phi1h.rename("phi1","phi1"); fileO.write(phi1h,time)
        phi2h.rename("phi2","phi2"); fileO.write(phi2h,time)
        ph.rename("p","p"); fileO.write(ph,time)

        for iterAdapt in range(adaptSteps):
            print("********* refinement step = ", iterAdapt)

            Adh = FunctionSpace(mesh, "DG", 0)
            wh = TestFunction(Adh)
            h = CellDiameter(mesh)
            he = FacetArea(mesh)

            ThetaT = wh*h**2*(div(grad(uh))+f)**2*dx
            globalT=assemble(ThetaT)
            globalT=globalT.get_local()
            globalT_max = max(globalT)
            Theta = sqrt(sum(globalT))
            print("error_estimate = ", Theta)

            if (Theta < tolAdapt and iterAdapt==adaptSteps):
                break
    
            markers = MeshFunction('bool', mesh, 2, False)
    
            for c in cells ( mesh ):
                markers[c] = globalT[c.index()] > (ref_ratio * globalT_max)

    
            mesh = refine(mesh, markers)
            adapt(bdry, mesh)
            mesh.smooth(10)
        
        
    inc += 1; time += dt
