addpath('../');
addpath '../PROPACK';
imgs = imgreader('image1/','*.jpg');
tem = imread('src/0000.jpg');
[x,y]=size(tem);
tem = reshape(tem,1,x*y)*1.0;
[length,n] = size(imgs);

r=18;
tol=0.0001;
% algorithm paramters
params.step_const = 0.8; % step size parameter for gradient descent
params.max_iter   = 1000;  % max number of iterations
params.tol        = tol;% stop when ||Y-UV'-S||_F/||Y||_F < tol
params.do_project = 0;
params.tos=5;
params.alpha=0.1;

t1 = tic;

for i =1:length
    img = reshape(imgs(i,:),x,y);
    [u,v,s]=rpca_nu(img, r, params);
    toc(t1)
    l=u*v';
    image_name=strcat('output2\X_image_nu',num2str(i));
    image_name=strcat(image_name,'.jpg');
    imwrite(mat2gray( reshape(l,x,y)),image_name);
    [u,c,v] = svd(img);
    l=u(:,1:r)*c(1:r,1:r);
    l=l*v(:,1:r)';
    image_name=strcat('output2\X_image_svd',num2str(i));
    image_name=strcat(image_name,'.jpg');
    imwrite(mat2gray( reshape(l,x,y)),image_name);
    
    
end