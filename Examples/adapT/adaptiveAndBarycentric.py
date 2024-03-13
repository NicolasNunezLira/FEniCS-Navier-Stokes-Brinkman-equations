import meshio

msh = meshio.read("TestMesh01.msh")
for cell in msh.cells:
    if cell.type == "triangle":
        triangle_cells = cell.data
    elif  cell.type == "tetra":
        tetra_cells = cell.data

meshio.write("TestMesh02.xdmf", meshio.Mesh(points=msh.points,
                                            cells=[("triangle", triangle_cells)]))        

from dolfin import *
import os


parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters['form_compiler']['quadrature_degree'] = 4

mesh = Mesh()
with XDMFFile("TestMesh02.xdmf") as infile:
    infile.read(mesh)

    
fileO = XDMFFile("out/testAdapt.xdmf")
fileO.parameters['rewrite_function_mesh']=False
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True


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
solve(a == L, uh, bc)
uh.rename("u","u"); fileO.write(uh,0.)


# then I comput something with that mesh and mabe refine it
# or something. Then store the mesh as msh
os.system('meshio-convert TestMesh02.xdmf TestMesh03.msh')

# then apply barycentric refinement
os.system('gmsh -barycentric_refine TestMesh03.msh -o TestMesh04.msh')

# then read again back into xdmf
msh2 = meshio.read("TestMesh04.msh")
for cell in msh2.cells:
    if cell.type == "triangle":
        triangle_cells = cell.data
    elif  cell.type == "tetra":
        tetra_cells = cell.data

meshio.write("TestMesh05.xdmf", meshio.Mesh(points=msh2.points,
                                            cells=[("triangle", triangle_cells)]))
