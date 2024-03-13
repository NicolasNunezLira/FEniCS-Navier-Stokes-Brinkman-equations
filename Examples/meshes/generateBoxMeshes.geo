// refinements generated one by one with
// gmsh boxGeometry.geo -3 -o box.msh
// then gmsh barycentric_refine box.msh
// then meshio-convert box.msh box.xml

h=1; // from 2 to 32
lc=1/h;
Point(1) = {0, 0, 0, lc};
Point(2) = {1., 0, 0, lc};
Point(3) = {1, 0.5, 0, lc};
Point(4) = {0, 0.5, 0, lc};
Point(11) = {0, 0, 0.5, lc};
Point(12) = {1., 0, 0.5, lc};
Point(13) = {1, 0.5, 0.5, lc};
Point(14) = {0, 0.5, 0.5, lc};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {11, 12};
Line(6) = {12, 13};
Line(7) = {13, 14};
Line(8) = {14, 11};
Line(9) = {11, 1};
Line(10) = {12, 2};
Line(11) = {13, 3};
Line(12) = {14, 4};

Line Loop(1) = {4, -9, -8, 12};
Plane Surface(1) = {1};

Line Loop(2) = {7, 12, -3, -11};
Plane Surface(2) = {2};

Line Loop(3) = {6, 11, -2, -10};
Plane Surface(3) = {3};

Line Loop(4) = {5, 10, -1, -9};
Plane Surface(4) = {4};

Line Loop(5) = {8, 5, 6, 7};
Plane Surface(5) = {5};

Line Loop(6) = {1, 2, 3, 4};
Plane Surface(6) = {6};

Surface Loop(1) = {5, 1, 6, 4, 3, 2};

Transfinite Surface {1,2,3,4,5,6};

Volume(1) = {1};
Transfinite Volume {1};
Mesh.ScalingFactor = 1./4.5;

//Mesh.Optimize = 10;


