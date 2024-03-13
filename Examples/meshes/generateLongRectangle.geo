h=100;
lc=2000/h;
Point(1) = {0, -1000, 0, lc};
Point(2) = {2000, -1000, 0, lc};
Point(3) = {2000, 0, 0, lc};
Point(4) = {0, 0, 0, lc};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(12) = {1, 2, 3, 4};
Plane Surface(13) = {12};
Transfinite Surface {13};
Mesh.ScalingFactor = 1./4.5;
//Mesh.Optimize = 10;
Mesh 2;
