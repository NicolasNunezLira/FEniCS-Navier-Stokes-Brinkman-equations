h= 0.2;
Point(1) = {-0.5, -0.5, 0, h};
Point(2) = {0.5, -0.5, 0, h};
Point(3) = {0.5, 0.5, 0, h};
Point(4) = {-0.5, 0.5, 0, h};
Point(5) = {0., 0.5, 0, h};// circle centre
Point(6) = {0., 1.0, 0, h};

Line(11) = {1, 2};
Line(12) = {2, 3};
Line(13) = {4, 1};

Circle(14) = {3,5,6};
Circle(15) = {6,5,4};

Curve Loop(16) = {12, 14, 15, 13, 11};
Plane Surface(17) = {16};



Field[1] = Distance;
Field[1].NNodesByEdge = 100;
Field[1].EdgesList = {16};
 
// Threshold
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = h / 2.;
Field[2].LcMax = h;
Field[2].DistMin = 0.1;
Field[2].DistMax = 0.9;

Mesh.CharacteristicLengthFactor = 0.2;
//Mesh.CharacteristicLengthExtendFromBoundary = 0;
Mesh.ScalingFactor = 1./4.5;
Mesh.Optimize=10;

Mesh 2;

Save Sprintf("tombStoneBaryFine.msh");
//Save Sprintf("unitSqBaryRef%03g.msh", 0);

