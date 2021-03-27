addpath('../');
addpath '../PROPACK';
imgs = imgreader('image1/','*.jpg');
tem = imread('src/0000.jpg');
[x,y]=size(tem);
tem = reshape(tem,1,x*y)*1.0;
[length,n] = size(imgs);

r=1;
tol=0.0001;
% algorithm paramters
params.step_const = 0.8; % step size parameter for gradient descent
params.max_iter   = 1000;  % max number of iterations
params.tol        = tol;% stop when ||Y-UV'-S||_F/||Y||_F < tol
params.do_project = 0;
params.tos=5;
params.alpha=0.5;

t1 = tic;   
[u,v,s]=rpca_nu(imgs, r, params);
toc(t1)
l=u*v';
count=0;
for i =1:length
    image_name=strcat('output1\X_image_nu',num2str(i));
    image_name=strcat(image_name,'.jpg');
    imwrite(mat2gray( reshape(l(i,:),x,y)),image_name);
    
    count=count+norm(double(l(i,:))-double(tem));
end
count/norm(double(tem))


t1 = tic; 
alpha_bnd=0.3;
[u,v]=rpca_gd(imgs, r, alpha_bnd, params);
toc(t1)

l=u*v';
count=0;
for i =1:length
    image_name=strcat('output1\X_image_gd',num2str(i));
    image_name=strcat(image_name,'.jpg');
    imwrite(mat2gray( reshape(l(i,:),x,y)),image_name);
    count=count+norm(double(l(i,:))-double(tem));
end
count/norm(double(tem))


t1 = tic; 
[u,v,g]=inexact_alm_rpca(imgs,0.005,tol,1000);
toc(t1)
l=u;
count=0;
for i =1:length
    image_name=strcat('output1\X_image_lm',num2str(i));
    image_name=strcat(image_name,'.jpg');
    imwrite(mat2gray( reshape(l(i,:),x,y)),image_name);
    count=count+norm(double(l(i,:))-double(tem));
end
count/norm(double(tem))

