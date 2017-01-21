function [A_dual, E_dual, dtau_dual, iter, Y] = RASL_inner_ialm(Dt, Jaco, lambda, tolerance, maxIter)
%--------------------------------------------------------------------------
% Xue Shengke, Zhejiang University, December 2016.
% Contact information: see readme.txt
%
% Partially composed of Peng Y., et al. RASL implementation, November 2009.
%--------------------------------------------------------------------------
%   RASL inner inexact ALM algorithm
%   
%   Inputs:
%       Dt           --- transformed batch image matrix, D o tau
%       Jaco         --- Jacobian matrix of Dt
%       lambda       --- parameters for ||E||_1
%       tolerance    --- stop iteration threshold
%       maxIter      --- maximum iteration
%
%   Outputs: 
%       A_dual       --- low-rank component
%       E_dual       --- sparse component
%       dtau_dual    --- delta transformation parameters
%       iter         --- number inner iterations
%       Y            --- lagrange multiplier
%--------------------------------------------------------------------------
% objective function:
% min ||A||_* + lambda ||E||_1 s.t. D o tau + sum(J*DeltaTau*eps*eps^T) = A + E
% Lagrangian function:
% L(A, E, DeltaTau, Y) = ||A||_* + lambda ||E||_1 
%     + <Y, D o tau + sum(J*DeltaTau*eps*eps^T) - A - E> 
%     + mu/2 ||D o tau + sum(J*DeltaTau*epsilon*epsilon^T) - A - E||_F^2
%--------------------------------------------------------------------------
[m, n] = size(Dt);

if nargin < 3
    error('Too few arguments');
end

if nargin < 4
    tolerance = 1e-7;
end

if nargin < 5
    maxIter = 1000;
end

DISPLAY_EVERY = 10 ;

Y = Dt;
norm_two = norm(Y, 2);
norm_inf = norm(Y(:), inf) / lambda;
dual_norm = max(norm_two, norm_inf);
Y = Y / dual_norm;

obj_value = Dt(:)' * Y(:);

A_dual = zeros(m, n);
E_dual = zeros(m, n);
dtau_dual = cell(n, 1);
dtau_dual_matrix = zeros(m, n);

mu = 1.25 / norm(Dt);
rho = 1.25;
mu_max = 1e+20;

d_norm = norm(Dt, 'fro');

iter = 0;
converged = false;
%% start optimization
while ~converged       
    iter = iter + 1;
    
    % A
    temp_T = Dt + dtau_dual_matrix - E_dual + (1/mu)*Y;
    temp_T(isnan(temp_T)) = 0;
    temp_T(isinf(temp_T)) = 0;
    [U, S, V] = svd(temp_T, 'econ');
    diagS = diag(S);
    A_dual = U * diag( pos(diagS-1/mu) ) * V';
    
    % E
    temp_T = Dt + dtau_dual_matrix - A_dual + (1/mu)*Y;
    E_dual = sign(temp_T) .* pos( abs(temp_T) - lambda/mu );
    
    % tau
    temp_T = A_dual + E_dual - Dt - (1/mu)*Y;
    for i = 1 : n
        dtau_dual{i} = Jaco{i}' * temp_T(:, i) ;
        dtau_dual_matrix(:, i) = Jaco{i} * dtau_dual{i} ;
    end
    
    % Y
    Z = Dt + dtau_dual_matrix - A_dual - E_dual;
    Y = Y + mu * Z;
    
    % mu
    mu = min(mu*rho, mu_max);    
    
    obj_value = Dt(:)' * Y(:); 
    
    stop_condition = norm(Z, 'fro') / d_norm;
    rank_A = rank(A_dual);
    L1_E = length(find(abs(E_dual)>0));
    
    if mod(iter, DISPLAY_EVERY) == 0
        disp(['#Iter ' num2str(iter) '  rank(A) ' num2str(rank_A) ...
            '  ||E||_0 ' num2str(L1_E) '  obj_value ' num2str(obj_value) ...
            '  stop condition ' num2str(stop_condition)]);
    end        
    
    if stop_condition <= tolerance
        disp('RASL inner loop is converged at:');
        disp(['#Iter ' num2str(iter) '  rank(A) ' num2str(rank_A) ...
            '  ||E||_0 ' num2str(L1_E) '  obj_value ' num2str(obj_value) ...
            '  stop condition ' num2str(stop_condition)]);
        converged = true ;
    end
    
    if ~converged && iter >= maxIter
        disp('Maximum iterations reaches') ;
        converged = true ;       
    end
end
