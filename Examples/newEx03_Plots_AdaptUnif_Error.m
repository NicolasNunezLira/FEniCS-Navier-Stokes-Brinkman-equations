clear all; %close all;

% AAA
%nn   &  hh  &   e(u)  &  r(u)  & e(t)  &  r(t)  &  e(sig) & r(sig) 
%==========================================================================
% BBB
%e(phi)  & r(phi)  & e(tj)  & r(tj)  & e(sigj) & r(sigj) & e(p) & r(p) & eff
%==========================================================================	     
	     
M1A=[1941    1  5.662     0  169.23     0  4225     0
7697       0.5  5.694  -0.0  149.05  -0.5  4175  0.0173
30657     0.25  6.295  -0.1  163.1  -0.7  6179  -0.5657
122369   0.125  3.302  0.93  166.7  -0.0  5012  -0.9827
488961  0.0625  1.5097  2.6  91.03  1.04  4139  -0.5513];

M1B = [0.5865 0  3.256     0  36.27     0 612.6     0 1.039
0.2256    2.224  1.66  0.9718  26.19  0.4698  258  1.248 1.341
0.19212  0.4468  1.012  0.7146  19.11  0.4547 131.6  0.9711 2.779
0.1771  0.9491  0.7259  0.4787  9.92  0.9461  157  -0.2548 0.742
0.10383  3.148  0.1246  2.543  4.504  2.722 118.3  0.408 4.731];



M2A=[2609  0.5286  3.398     0    128.5     0  4290     0
9979       0.5     2.997  -0.242  61.3  -1.129  2667  -0.5212
25691  0.345       1.075  0.5544  40  -0.5636  1809  -1.831
46866  0.3411      0.527  2.942  21.8  3.335   1123  -1.478
85963  0.3377      0.2971  6.143  16.88  1.817   679  1.555
192386  0.338      0.17397  2.433  9.53  0.5333  155  1.639];

M2B=[0.1662 0  1.539     0  19.25     0       420.5     0   2.114
0.04683  1.888  0.6336  1.323  14.47  -0.1627 172.6  1.328  2.1299
0.01704  0.4962  0.1343  0.3605  7.181  2.316 102.8  0.2572 2.1075
0.00996  2.057  0.06251  2.513  3.187  2.703   73.2  0.4574 2.1966
0.003648  5.603  0.03998  6.057  0.4731  6.289 40.01  2.12  2.217
0.000838  0.374  0.01656  2.188  0.1729  2.498 26.39  1.624 2.207];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n1 = M1A(:,1); n2=M2A(:,1); h1 = M1A(:,2);

phi1=M1B(:,1); p1=M1B(:,7); u1 = M1A(:,3);
phi2=M2B(:,1); p2=M2B(:,7); u2 = M2A(:,3);

t1 = M1A(:,5); sig1 = M1A(:,7);
t1j = M1B(:,3); sig1j = M1B(:,5); ef1=M1B(:,9);
t2 = M2A(:,5); sig2 = M2A(:,7);
t2j = M2B(:,3); sig2j = M2B(:,5); ef2=M2B(:,9);

HH1=figure('Position',[429 254 500 500]);
hh1=axes('Position',[0.1 0.1 0.86 0.86]);

G1=loglog(n1,h1.^2,'ko-',n1,u1,'*--',n2,u2,'*-',n1,t1,'d--',n2,t2,'d-',n1,sig1,'v--',n2,sig2,'v-',n1,p1,'p--',n2,p2,'p-');
set(G1,'Linewidth',1.5);
set(G1,'Markersize',10);
xlabel('DoF','Fontsize',20,'Interpreter','LaTex');
grid on; 
legend('$O(h^2)$','$e(\mathbf{u})$ - uniform','$e(\mathbf{u})$ - adaptive', ...
       '$e(\mathbf{t})$ - uniform','$e(\mathbf{t})$ - adaptive', ...
       '$e(\mathbf{\sigma})$ - uniform','$e(\mathbf{\sigma})$ - adaptive', ...
       '$e(p)$ - uniform','$e(p)$ - adaptive');
set(gca,'Linewidth',1.0);
set(gca,'TickLabelInterpreter','latex')
set(gca,'Fontsize',16); 


HH2=figure('Position',[429 354 500 500]);
hh2=axes('Position',[0.1 0.1 0.86 0.86]);

G2= loglog(n1,h1.^2,'ko-',n1,phi1,'s--',n2,phi2,'s-',n1,t1j,'+--',n2,t2j,'+-',n1,sig1j,'^--',n2,sig2j,'^-');
set(G2,'Linewidth',1.5);
set(G2,'Markersize',10);
xlabel('DoF','Fontsize',20,'Interpreter','LaTex'); 
grid on; 
legend('$O(h^2)$','$e(\varphi)$ - uniform', '$e(\varphi)$ - adaptive', ...
       '$e(\tilde{\mathbf{t}})$ - uniform', '$e(\tilde{\mathbf{t}})$ - adaptive', ...
       '$e(\tilde{\mathbf{\sigma}})$ - uniform', '$e(\tilde{\mathbf{\sigma}})$ - adaptive');
set(gca,'Linewidth',1.0);
set(gca,'TickLabelInterpreter','latex')
set(gca,'Fontsize',16); 

HH2=figure('Position',[229 354 500 500]);
hh2=axes('Position',[0.1 0.1 0.86 0.86]);
G3 = semilogx(n1,ef1,'o--',n2,ef2,'o-');
set(G3,'Linewidth',1.5); 
set(G3,'Markersize',10);

legend('eff$(\Psi)$ - uniform','eff$(\Psi)$ - adaptive');
xlabel('DoF','Fontsize',20,'Interpreter','LaTex');
grid on; 
set(gca,'Linewidth',1.0);
set(gca,'TickLabelInterpreter','latex')
set(gca,'Fontsize',16); 
