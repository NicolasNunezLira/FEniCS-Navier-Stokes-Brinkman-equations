h=2;
lc=2/h;
Point(1) = {-1, -1, 0, lc};
Point(2) = {1, -1, 0, lc};
Point(3) = {1, 1, 0, lc};
Point(4) = {-1, 1, 0, lc};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(12) = {1, 2, 3, 4};
Plane Surface(13) = {12};
Transfinite Surface {13};
Mesh.ScalingFactor = 1./4.5;

Mesh 2;

Save Sprintf("unitSqRef%03g.msh",0);
Save Sprintf("unitSqBaryRef%03g.msh", 0);
// and then one should do per each of these gmsh -barycentric_refine unitSqBaryRef%03g.msh

For i In {1:6}
  RefineMesh;
  Save Sprintf("unitSqRef%03g.msh", i);
  Save Sprintf("unitSqBaryRef%03g.msh", i);
EndFor