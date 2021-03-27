addpath('../');
addpath '../PROPACK';

m=1000;
n=1000;
r  = 1; 
tol=0.01;
M= randm(r,m,n);
R =round(full(sprand(m,n,0.1))*6);
% A=ones(m,n);

c=M+R;



% algorithm paramters
params.step_const = 1; % step size parameter for gradient descent
params.max_iter   = 1000;  % max number of iterations
params.tol        = tol;% stop when ||Y-UV'-S||_F/||Y||_F < tol
params.do_project = 0;
params.tos=5;
params.alpha=0.05;


t1 = tic;   
[u,v,s]=rpca_nu(c, r, params);
toc(t1)
l=u*v';
s(1:10,1:10)
R(1:10,1:10)
R(1:10,1:10)-s(1:10,1:10)
M(1:10,1:10)-l(1:10,1:10)
function M = randm(r,m,n)
M=zeros(m,n);
b=ones(m,n);
b=tril(b);
for i=1:1:m;
    v=round(rand(1,r));
    u=v*b(end-r+1:end,:);
    M(i,:)=u;
end
end