from dolfin import *
import os


parameters["form_compiler"]["representation"] = "uflacs"
parameters["refinement_algorithm"] = "plaza_with_parent_facets" # for adaptive!
parameters["form_compiler"]["cpp_optimize"] = True
parameters['form_compiler']['quadrature_degree'] = 4

mesh = RectangleMesh(Point(0,-1000),Point(2000,0),16,8)
    
fileO = XDMFFile("out/testAdapt.xdmf")
fileO.parameters['rewrite_function_mesh']=True # for adaptive!
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True


# ******** Solve-Estimate-Mark-Refine loop **** #
adaptSteps = 5
tolAdapt = 1.0E-6
ref_ratio = 0.1  

for iterAdapt in range(adaptSteps):

    print("********* refinement step = ", iterAdapt)
    meshname="out/refinedMesh%01g.xdmf"%iterAdapt
    
    
    bdry = MeshFunction("size_t", mesh, 1)
    bdry.set_all(0)
    
    '''
    class Inlet(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.)  and on_boundary

    class Outlet(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 6.)  and on_boundary

    class Wall(SubDomain):
        def inside(self, x, on_boundary):
            return (near(x[0], 1) or near(x[1],1) or near(x[1],0.) or near(x[1],2.)) and on_boundary

    Inlet().mark(bdry,inlet)
    Outlet().mark(bdry,outlet)
    Wall().mark(bdry,wall)
    '''

    ds = Measure("ds", subdomain_data=bdry)
    dx = Measure("dx", domain=mesh)
    dS = Measure("dS", subdomain_data=bdry)

    # ********* Finite dimensional spaces ********* #
    P1  = FiniteElement("CG", mesh.ufl_cell(), 1)
    Vh = FunctionSpace(mesh, P1)

    bc = DirichletBC(Vh, Constant(0.), 'on_boundary')
                 
    u = TrialFunction(Vh)
    v = TestFunction(Vh)
    f = Expression("1e5*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=1)
    g = Expression("10*sin(5*x[0])", degree=1)
    a = inner(grad(u), grad(v))*dx
    L = f*v*dx + g*v*ds

    uh = Function(Vh)

    # ******* Solving *************** #
    solve(a == L, uh, bc)
    counting = 1.0* iterAdapt
    uh.rename("u","u"); fileO.write(uh,counting)

    # ********* Error estimation and mesh adaptativity ******** #
    
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

    with XDMFFile(meshname) as infile:
        infile.write(mesh)

    # then I comput something with that mesh and mabe refine it
    # or something. Then store the mesh as msh
    os.system('meshio-convert out/refinedMesh%01g.xdmf out/refinedMesh%01g.msh'%(iterAdapt,iterAdapt))

    # then apply barycentric refinement
    os.system('gmsh -barycentric_refine out/refinedMesh%01g.msh -o out/baryRefinedMesh%01g.msh'%(iterAdapt,iterAdapt))
