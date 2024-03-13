from dolfin import *
import meshio

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters['form_compiler']['quadrature_degree'] = 4

#mesh =  Mesh("zpBaryUnstr.xml")
mesh =  Mesh("unitSqBaryRef004.xml")
#mesh =  Mesh("unitSqRef000.xml")
#mesh =  Mesh("tombStone.xml")
#mesh =  Mesh("tombStoneBaryFine.xml")
#mesh =  Mesh("boxRef004.xml")
fileO = XDMFFile(mesh.mpi_comm(), "out/test.xdmf")
fileO.parameters['rewrite_function_mesh']=False
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True

# ********* Finite dimensional spaces ********* #
P1  = FiniteElement("CG", mesh.ufl_cell(), 1)

Vh = FunctionSpace(mesh, P1)

el = Function(Vh)

u_h = interpolate(Constant(1.),Vh)

u_h.rename("u","u"); fileO.write(u_h,0.)
