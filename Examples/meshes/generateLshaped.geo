h=2;
lc=2/h;
Point(1) = {-1, -1, 0, lc};
Point(2) = {1, -1, 0, lc};
Point(3) = {1, 0, 0, lc};
Point(4) = {0, 0, 0, lc};
Point(5) = {0, 1, 0, lc};
Point(6) = {-1,1,0,lc};

Line(11) = {1, 2};
Line(12) = {2, 3};
Line(13) = {3, 4};
Line(14) = {4, 5};
Line(15) = {5, 6};
Line(16) = {6,1};

Line Loop(22) = {11, 12, 13, 14,15,16};
Plane Surface(23) = {22};
Transfinite Surface {1,2,3,4,5,6};
Mesh.ScalingFactor = 1.;

Mesh 2;

Save Sprintf("LshapedRef%03g.msh",0);
Save Sprintf("LshapedBaryRef%03g.msh", 0);

// ##### and then one should do per each of these gmsh -barycentric_refine unitSqBaryRef%03g.msh

For i In {1:6}
  RefineMesh;
  Save Sprintf("LshapedRef%03g.msh", i);
  Save Sprintf("LshapedBaryRef%03g.msh", i);
EndFor