%% video RobustPCA example: separates background and foreground
addpath('../');

% ! the movie will be downloaded from the internet !
movieFile = 'httpcvrr-ucsd-edu.avi';
% movieFile = 'RobustPCA_video_demo.AVI';
% urlwrite('https://github.com/dlaptev/RobustPCA/blob/master/examples/RobustPCA_video_demo.avi?raw=true', movieFile);

% open the movie
n_frames = 180;
movie = VideoReader(movieFile);
frate = movie.FrameRate;    
height = movie.Height;
width = movie.Width;

% vectorize every frame to form matrix X
X = zeros(n_frames, height*width);
for i = (1:n_frames)
    frame = read(movie, i);
    frame = rgb2gray(frame);
    X(i,:) = reshape(frame,[],1);
end

% apply Robust PCA
lambda = 1/sqrt(max(size(X)));
tic

% [L,S] = RPCA(X, lambda/3, 10*lambda/3, 1e-5);
r  = 1; 
p  = 1;

% algorithm paramters
params.step_const = 1; % step size parameter for gradient descent
params.max_iter   = 2000;  % max number of iterations
params.tol        = 0.02;% stop when ||Y-UV'-S||_F/||Y||_F < tol
params.do_project =0;
params.tos =5;
params.alpha =1.2;
% alpha_bnd is some safe upper bound on alpha, 
% that is, the fraction of nonzeros in each row of S (can be tuned)

alpha_bnd = 0.12; 

[U,V,S]=rpca_nu(X,r,params);
% [U,V,S]=rpca_gd(X, r, alpha_bnd, params);
L=U*V';
S=X-L;
% 
% [L,S,x]=inexact_alm_rpca(X,0.0007,0.02,1000);
% S=X-L;
toc

% prepare the new movie file
vidObj = VideoWriter('RobustPCA_video_output.avi');
vidObj.FrameRate = frate;
open(vidObj);
range = 255;
map = repmat((0:range)'./range, 1, 3);
S = medfilt2(S, [5,1]); % median filter in time
for i = (1:size(X, 1));
    frame1 = reshape(X(i,:),height,[]);
    frame2 = reshape(L(i,:),height,[]);
    frame3 = reshape(abs(S(i,:)),height,[]);
    % median filter in space; threshold
    frame3 = (medfilt2(abs(frame3), [5,5]) > 5).*frame1;
    % stack X, L and S together
    frame = mat2gray([frame1, frame2, frame3]);
    frame = gray2ind(frame,range);
    frame = im2frame(frame,map);
    writeVideo(vidObj,frame);
%     if mod(i,10) ==0
%     image_name=strcat('C:\Users\LENOVO\Desktop\RPCA_GD\Image\X_image_gd',num2str(i));
%     image_name=strcat(image_name,'.jpg');
%     imwrite(mat2gray(frame1),image_name,'jpg'); 
%     image_name=strcat('C:\Users\LENOVO\Desktop\RPCA_GD\Image\L_image_gd',num2str(i));
%     image_name=strcat(image_name,'.jpg');
%     imwrite(mat2gray(frame2),image_name,'jpg');
%     image_name=strcat('C:\Users\LENOVO\Desktop\RPCA_GD\Image\S_image_gd',num2str(i));
%     image_name=strcat(image_name,'.jpg');
%     imwrite(mat2gray(frame3),image_name,'jpg'); 
%     end
    end
close(vidObj);