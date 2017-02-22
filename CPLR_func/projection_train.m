function [W, B, iter] = projection_train(D, A, idx, para, destDir)
%--------------------------------------------------------------------------
% Xue Shengke, Zhejiang University, December 2016.
% Contact information: see readme.txt
%
%--------------------------------------------------------------------------
%     train a fast prjection function from original data to low-rank data
% 
%     Inputs:
%         D            --- transformed batch image matrix, D o tau
%         A            --- low-rank representation
%         idx          --- index of class
%         para         --- parameters struct
%         destDir      --- save directory
% 
%     Outputs: 
%         W            --- weight
%         B            --- bias
%         iter         --- number inner iterations
%--------------------------------------------------------------------------
classNum = para.classNum;
%% reconstruct and normalize D and A 
for i = 1 : classNum
    D{i} = D{i}.D ./ repmat(max(D{i}.D), [size(D{i}.D,1) 1]);    
%     A{i} = A{i}.A ./ repmat(max(A{i}.A), [size(A{i}.A,1) 1]);

    Am = mean(A{i}.A, 2);
    Am = Am / max(Am);
    A{i} = repmat(Am, [1 size(A{i}.A,2)]);  
end

%% initialize the variables
[m, n] = size(D{idx});
W = 0.5 * rand(m) / sqrt(m); % initial value of weight
B = zeros(m, n);

DISPLAY_EVERY = 10 ;    % display values interval

alpha = para.alpha;
gamma = para.gamma;
beta = para.beta;
lr = para.lr;
eta = lr(1);
threshold = para.proj_threshold;
maxIter = para.proj_maxIter;
iter = 0;
pre_obj = Inf;
cur_obj = 0;
delta = abs(pre_obj - cur_obj);

actv = para.actv_func;
der_actv = para.der_actv_func;

vW = zeros(size(W));
vB = zeros(size(B));

F  = cell(classNum, 1);
dF = cell(classNum, 1);
for i = 1: classNum
    F{i}  = actv(W * D{i} + B);
    dF{i} = der_actv(F{i});
end

%% iterative optimization starts
converged = false;
while ~converged
    iter = iter + 1;
    
%     if iter > 200
    if     delta < 1000 * threshold, eta = lr(2);
    elseif delta < 100 * threshold , eta = lr(3);
    elseif delta < 10 * threshold  , eta = lr(4);
    end   
%     end
    
    tempW = 0;
    for i = 1 : classNum
        tempW = tempW + indicator(idx,i,beta) * dF{i} .* (F{i} - A{idx}) * D{i}' ;             
    end
    dW = tempW + gamma * W;
    vW = alpha * vW + eta * dW;
    W = W - vW;
    
    tempB = 0;
    for i = 1 : classNum
        tempB = tempB + indicator(idx,i,beta) * dF{i} .* (F{i} - A{idx}) ; 
    end
    tempB = repmat(mean(tempB,2), [1, size(tempB,2)]);
    dB = tempB + gamma * B;
    vB = alpha * vB + eta * dB;
    B = B - vB;
    
    for i = 1: classNum
        F{i}  = actv(W * D{i} + B);
        dF{i} = der_actv(F{i});
    end
    
%     residual = 0;
%     for i = 1 : classNum
%         residual = residual + indicator(idx,i,beta) * norm(A{idx} - F{i}, 'fro');
%     end
    residual = norm(A{idx} - F{idx}, 'fro');
    regular  = norm(W,'fro') + norm(B,'fro');
%     cur_obj = (1/2) * residual + (gamma/2) * regular;
    cur_obj = (1/2) * residual;

    if mod(iter, DISPLAY_EVERY) == 0
    disp(['#Iter ' num2str(iter) '  ||A-f(WD+B)||^2 ' num2str(residual) ...
        '  ||W||^2+||B||^2 ' num2str(regular) '  objvalue ' num2str(cur_obj)]);
    end        
    
    delta = abs(pre_obj - cur_obj);
    if ( delta < threshold || iter >= maxIter )
        converged = true;
        if delta >= threshold
            disp('Maximum iterations reaches') ;
        else
            disp('Projection train converges');
        end
        disp(['#Iter ' num2str(iter) '  ||A-f(WD+B)||^2 ' num2str(residual) ...
            '  ||W||^2+||B||^2 ' num2str(regular) '  objvalue ' num2str(cur_obj)]);
    else
        pre_obj = cur_obj;
    end
end

%% save the peojection parameters
save(fullfile(destDir, 'projection.mat'), 'W', 'B');

%% display training results
% figure;
% subplot(2,2,1);
% grid_plot(W, size(W,2), para.imageSize); title('W');
% subplot(2,2,2);
% % grid_plot(B, size(B,2), para.imageSize); title('B');
% subplot(2,2,3);
% grid_plot(sigmoid(W*D{idx}), size(D{idx},2), para.imageSize); title('activation');
% subplot(2,2,4);
% grid_plot(A{idx}-sigmoid(W*D{idx}), size(D{idx},2), para.imageSize); title('residual');

if para.DISPLAY 
    xI = ceil(sqrt(classNum));
    yI = ceil(classNum / xI);
    figure;
    for i = 1 : classNum
        subplot(xI,yI,i);
        grid_plot(sigmoid(W*D{i}), size(D{i},2), para.imageSize); 
        title(num2str(i));
        hold on;
    end
    hold off;
end
