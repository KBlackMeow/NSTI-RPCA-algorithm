function [U,V,S] = rpca_new(Y, r, params)
% [U, V] = RPCA_GD(Y, r, alpha, params)
% Robust PCA via Non-convex Gradient Descent
%
% Y : A sparse matrix to be decomposed into a low-rank matrix M and a sparse
% matrix S. Unobserved entries are represented as zeros.
% r : Target rank
% alpha : An upper bound of max sparsity over the columns/rows of S
% params : parameters for the algorithm
%   .step_const : Constant for step size (default .5)
%   .max_iter : Maximum number of iterations (default 30)
%   .tol : Desired Frobenius norm error (default 2e-4)
%   .incoh : Incoherence of M (default 5)
%
% Output:
% U, V : M=U*V' is the estimated lowrank matrix
%
% By:
% Xinyang Yi, Dohyung Park, Yudong Chen, Constantine Caramanis
% {yixy,dhpark,constantine}@utexas.edu, yudong.chen@cornell.edu


% Default parameter settings
step_const = .5;
max_iter   = 30;
tol        = 0.2;
tos=1;
alpha=1;
% Read paramter settings

if isfield(params,'step_const') step_const = params.step_const; end
if isfield(params,'max_iter')   max_iter = params.max_iter; end
if isfield(params,'tol')        tol= params.tol; end
if isfield(params,'tos') tos = params.tos; end
if isfield(params,'alpha') alpha = params.alpha; end

% Library paths
addpath PROPACK;
% addpath MinMaxSelection;

% Setting up
err  = zeros(1,max_iter);
time = zeros(1,max_iter);
Ynormfro = norm(Y,1);
[d1, d2] = size(Y);

is_sparse  = issparse(Y);
if is_sparse
    [I, J, Y_vec] = find(Y);
    n = length(Y_vec);
    obs_ind = sub2ind([d1,d2], I, J);
    col = [0; find(diff(J)); n];
    p = n/d1/d2;
    if p>0.9
        is_sparse = 0;
        Y = full(Y);
    end
else
    p = 1;
end

%% Phase I: Initialization
t1 = tic; t = 1;

% Initial sparse projection
fprintf('Initial sparse projection; time %f \n', toc(t1));

% Initial factorization
fprintf('Initial SVD; time %f \n', toc(t1));

[U,Sig,V]=lansvd((Y)/p,r,'L');
U = U(:,1:r) * sqrt(Sig(1:r,1:r));
V = V(:,1:r) * sqrt(Sig(1:r,1:r));

% Compute the initial error
err(t)  = inf;

time(t) = toc(t1);

%% Phase II: Gradient Descent
steplength = step_const / sqrt(Sig(1,1));
if is_sparse
    YminusUV = sparse(I, J, 1, d1, d2, n);
else
    YminusUV = zeros(d1, d2);
end

fprintf('Begin Gradient descent\n');
converged = 0;
while ~converged
    
    t = t + 1;
    
    %%
    if is_sparse
        UVobs_vec = compute_X_Omega(U, V, obs_ind);
        %UVobs_vec = partXY(U', V', I, J, n)';
        YminusUV = sparse(I, J, Y_vec-UVobs_vec, d1, d2, n); clearvars UVobs_vec;
    else
        YminusUV = Y - U*V';
    end
    
    %% Sparse Projection for S
    tos=sum(abs(YminusUV(:)))/(d1*d2)*alpha;

    S=sign(YminusUV) .* max(abs(YminusUV) - tos, 0);


    E = YminusUV - S;
    
%     err(t) = (tos*norm(S, 1)+0.5*norm(E, 'fro'))/Ynormfro;
     err(t) = (norm(S, 1)+norm(E, 'fro'))/Ynormfro;
%     clearvars S;
    
% ???? 

%     hess = V'*V;
%     hess = inv(hess);
%     Unew = U +  steplength *( (E * V) /p)*hess;
%     hess = U'*U;
%     hess = inv(hess);
%     Vnew = V + steplength *( (U' * E)' /p)*hess;
%  ?????
% 
    hess = V'*V;
    hess = inv(hess);
    Unew = U +  ( (E * V) /p)*hess;
    hess = U'*U;
    hess = inv(hess);
    Vnew = V + ( E'*U /p)*hess;
   
    U = Unew;
    V = Vnew;
% 
%     hess = V'*V;
%     hess = inv(hess);
%     U =U +  hess*( (E * V) /p);
%     hess = U'*U;
%     hess = inv(hess);
%     V = V + hess*( E'*U /p);

    
    %% Compute error
%     err(t) = norm(E, 'fro')/Ynormfro;
    
    time(t) = toc(t1);
    
    %% Convergence check
    fprintf('Iter no. %d err %e time %f \n', t, err(t), time(t));
    if (t >= max_iter)
        converged = 1;
        fprintf('Maximum iterations reached.\n');
    end
    if (err(t) <= max(tol,eps))
        converged = 1;
        fprintf('Target error reached.\n');
    end
    if (abs(err(t-1)/ err(t) -1) < 0.00001)
       converged = 1;
       fprintf('No improvement.\n');
    end
    
end

err  = err(1:t);
time = time(1:t);
