addpath('../');
addpath '../PROPACK';
a=load('Yale_64x64.mat');
[x,y]=size(a.fea);
imgs = a.fea;
imgs = imgs(1:11,:);
x=11;
img = zeros(64,64*11);
for i =1:x
    image_name=strcat('output\',num2str(i),'image_a');
    image_name=strcat(image_name,'.jpg');
    imwrite(mat2gray( reshape(imgs(i,:),64,64)),image_name);
    img(1:64,(i-1)*64+1:64*i)=reshape(imgs(i,:),64,64);
end
% imshow(mat2gray(img))
r=2;
tol=0.0001;
% algorithm paramters
params.step_const = 1; % step size parameter for gradient descent
params.max_iter   = 1000;  % max number of iterations
params.tol        = tol;% stop when ||Y-UV'-S||_F/||Y||_F < tol
params.do_project = 0;
params.tos=5;
params.alpha=0.01;

t1 = tic;   
[u,v,s]=rpca_nu(imgs, r, params);
toc(t1)
l=u*v';
img = zeros(64,64*11);
for i =1:x
    image_name=strcat('output\',num2str(i),'image_b');
    image_name=strcat(image_name,'.jpg');
    imwrite(mat2gray( reshape(l(i,:),64,64)),image_name);
    img(1:64,(i-1)*64+1:64*i)=reshape(l(i,:),64,64);
end
% imshow(mat2gray(img))
t1 = tic;
alpha_bnd=0.1;
[u,v]=rpca_gd(imgs, r, alpha_bnd, params);
toc(t1)
l=u*v';
img = zeros(64,64*11);
for i =1:x
    image_name=strcat('output\',num2str(i),'image_c');
    image_name=strcat(image_name,'.jpg');
    imwrite(mat2gray( reshape(l(i,:),64,64)),image_name);
    img(1:64,(i-1)*64+1:64*i)=reshape(l(i,:),64,64);
end
% imshow(mat2gray(img))
% 
t1 = tic;
lambda = 1/sqrt(1*max(x,y));
opts.tol = 1e-8;
opts.mu = 1e-4;
opts.rho = 1.1;
opts.DEBUG = 1;
[Lhat,Shat] = trpca_tnn(imgs,lambda,opts);
l=Lhat;
toc(t1)
img = zeros(64,64*11);
for i =1:x
    image_name=strcat('output\',num2str(i),'image_d');
    image_name=strcat(image_name,'.jpg');
    imwrite(mat2gray( reshape(l(i,:),64,64)),image_name);
    img(1:64,(i-1)*64+1:64*i)=reshape(l(i,:),64,64);
end
% imshow(mat2gray(img))
t1 = tic; 
[u,v,g]=inexact_alm_rpca(imgs,0.01,tol,1000);
toc(t1)
l=u;
img = zeros(64,64*11);
for i =1:x
    image_name=strcat('output\',num2str(i),'image_e');
    image_name=strcat(image_name,'.jpg');
    imwrite(mat2gray( reshape(l(i,:),64,64)),image_name);
    img(1:64,(i-1)*64+1:64*i)=reshape(l(i,:),64,64);
end
imshow(mat2gray(img))
img = zeros(64*5,64*11);
for i =1:x
    image_name=strcat('output\',num2str(i),'image_a');
    image_name=strcat(image_name,'.jpg');
    imga = imread([image_name]);
    image_name=strcat('output\',num2str(i),'image_b');
    image_name=strcat(image_name,'.jpg');
    imgb = imread([image_name]);
    b=psnr(imga,imgb);
    image_name=strcat('output\',num2str(i),'image_c');
    image_name=strcat(image_name,'.jpg');
    imgc = imread([ image_name]);
    c=psnr(imga,imgc);
    image_name=strcat('output\',num2str(i),'image_d');
    image_name=strcat(image_name,'.jpg');
    imgd = imread([image_name]);
    d=psnr(imga,imgd);
    image_name=strcat('output\',num2str(i),'image_e');
    image_name=strcat(image_name,'.jpg');
    imge = imread([image_name]);
    e=psnr(imga,imge);
    img(1:64,(i-1)*64+1:64*i)=imga;
    img(64+1:64*2,(i-1)*64+1:64*i)=imgb;
    img(64*2+1:64*3,(i-1)*64+1:64*i)=imgc;
    img(64*3+1:64*4,(i-1)*64+1:64*i)=imgd;
    img(64*4+1:64*5,(i-1)*64+1:64*i)=imge;
    fprintf("%f %f %f %f\n",b,c,d,e);
end
% imshow(mat2gray(img))
