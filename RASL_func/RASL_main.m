function [ Di, Do, A, E, tau, numIterOuter, numIterInner ] = RASL_main( ...
    batchImages, transformations, para, destDir)
%--------------------------------------------------------------------------
% Xue Shengke, Zhejiang University, December 2016.
% Contact information: see readme.txt
%
% Partially composed of Peng Y., et al. RASL implementation, November 2009.
%--------------------------------------------------------------------------
%     RASL main
% 
%     Inputs:
%         batchImages       --- train images
%         transformations   --- initial transformation
%         para              --- parameters for RASL
%         destDir           --- save directory
% 
%     Outputs: 
%         Di                --- input original images
%         Do                --- output aligned images
%         A                 --- low-rank component
%         E                 --- sparse error
%         tau               --- transformation parameters
%         numIterOuter      --- number of outer loop iterations
%         numIterInner      --- total number of inner loop iterations
%--------------------------------------------------------------------------
%% read and store full images
sigma0 = 2/5 ;
sigmaS = 1 ;
numImages = para.trainImageNum;

I0  = cell(para.numScales, numImages) ; % images
I0x = cell(para.numScales, numImages) ; % image derivatives
I0y = cell(para.numScales, numImages) ; % image derivatives

for i = 1 : numImages
    if ndims(batchImages) == 4
        currentImage = rbg2gray(batchImages(:, :, :, i));
    elseif ndims(batchImages) == 3
        currentImage = batchImages(:, :, i);
    end

    currentImagePyramid = gauss_pyramid( currentImage, para.numScales, ...
        sqrt(det(transformations{i}(1:2,1:2)))*sigma0, sigmaS );
        
    for scaleIdx = para.numScales : -1 : 1
        I0{scaleIdx, i} = currentImagePyramid{scaleIdx};
        
        % image derivatives, for Jacobian matrix
        I0_smooth = I0{scaleIdx, i};
        I0x{scaleIdx, i} = imfilter( I0_smooth, (-fspecial('sobel')') / 8 );
        I0y{scaleIdx, i} = imfilter( I0_smooth,  -fspecial('sobel')   / 8 );
    end   
end

%% get the initial input images in standard frame

imgSize = para.imageSize; 
tau_init = cell(numImages, 1); 
D = zeros(imgSize(1)*imgSize(2), numImages);

for i = 1 : numImages
    if size(transformations{i}, 1) < 3
        transformations{i} = [transformations{i} ; 0 0 1] ;
    end
    tau_init{i} = projective_matrix_to_parameters(para.transformType, transformations{i});

    % transformed image
    Tfm = fliptform(maketform('projective', transformations{i}'));           
    y   = vec(imtransform(I0{1,i}, Tfm, 'bicubic', 'XData', [1 imgSize(2)], 'YData', [1 imgSize(1)], 'Size', imgSize));
    y   = y / norm(y) ;
    D(:, i) = y ;
end
Di = D;     % as output parameter

%% save inital D and tau
if para.saveStart
    save(fullfile(destDir, 'original.mat'), 'D', 'tau_init') ;
end

%% start the main loop
T_in = cell(1,numImages) ;
T_ds = [ 0.5,   0, -0.5; ...
         0,   0.5, -0.5   ];
T_ds_hom = [ T_ds; [ 0 0 1 ]];

numIterOuter = 0 ; 
numIterInner = 0 ;
lambda = para.lambdac/sqrt(size(D,1)); 

tic
%% multi-scale optimization
for scaleIndex = para.numScales : -1 : 1
    iterNum = 0 ; 
    converged = false ;
    prevObj = inf ;    
    imgSize = para.imageSize / 2^(scaleIndex-1) ;    
    tau = cell(numImages, 1) ;

    % compute the transformations from last scale images 
    for i = 1 : numImages           
        if scaleIndex == para.numScales
            T_in{i} = T_ds_hom^(scaleIndex-1)*transformations{i}*inv(T_ds_hom^(scaleIndex-1)) ;
        else
            T_in{i} = inv(T_ds_hom)*T_in{i}*T_ds_hom ;
        end       
    end

    %% RASL main loop
    while ~converged

        iterNum = iterNum + 1 ;
        numIterOuter = numIterOuter + 1 ;
        
        Dt = zeros(imgSize(1)*imgSize(2), numImages);
        Jaco = cell(numImages, 1) ;
        disp(['scale ' num2str(scaleIndex) ', iter ' num2str(iterNum)]) ;
        
        for i = 1 : numImages
            % transformed image and derivatives 
            Tfm = fliptform(maketform('projective',T_in{i}'));            
            y   = vec(imtransform(I0{scaleIndex, i}, Tfm, 'bicubic', 'XData', [1 imgSize(2)], 'YData', [1 imgSize(1)], 'Size', imgSize));
            Iu  = vec(imtransform(I0x{scaleIndex,i}, Tfm, 'bicubic', 'XData', [1 imgSize(2)], 'YData', [1 imgSize(1)], 'Size', imgSize));
            Iv  = vec(imtransform(I0y{scaleIndex,i}, Tfm, 'bicubic', 'XData', [1 imgSize(2)], 'YData', [1 imgSize(1)], 'Size', imgSize));

            Iu = (1/norm(y))*Iu - ( (y'*Iu)/(norm(y))^3 )*y ;
            Iv = (1/norm(y))*Iv - ( (y'*Iv)/(norm(y))^3 )*y ;

            y = y / norm(y) ; 
            Dt(:, i) = y ;

            % transformation matrix to parameters
            tau{i} = projective_matrix_to_parameters(para.transformType, T_in{i}) ; 
            
            % Compute Jacobian
            Jaco{i} = image_Jaco(Iu, Iv, imgSize, para.transformType, tau{i}) ;
        end

       %% RASL inner loop -----------------------------------------
        % use QR decomposition to orthogonalize the Jacobian matrix
        for i = 1 : numImages
            [Q{i}, R{i}] = qr(Jaco{i}, 0);
        end
        
        [A, E, delta_tau, numIterInnerEach] = RASL_inner_ialm(Dt, Q, lambda, para.inner_tolerance, para.inner_maxIter);
        
        % paramters update by step
        for i = 1 : numImages
            delta_tau{i} = inv(R{i}) * delta_tau{i} ;
            tau{i} = tau{i} + delta_tau{i};
            T_in{i} = parameters_to_projective_matrix(para.transformType, tau{i});
        end
       %% ---------------------------------------------------------
       
        numIterInner = numIterInner + numIterInnerEach ;

        curObj = norm(svd(A), 1) + lambda*norm(E(:), 1);
        disp(['previous objective ' num2str(prevObj) ]);
        disp([' current objective ' num2str(curObj)  ]);
        
        % save intermedia results
        if para.saveIntermedia
            matName = ['scale_', num2str(scaleIndex),'_iter_', num2str(iterNum), ...
                '_outer_', num2str(numIterOuter), '_inner_', num2str(numIterInner), '.mat'] ;
            save(fullfile(destDir, matName),'Dt','A','E','tau') ;
        end
        
        if ( abs(prevObj - curObj) < para.outer_tolerance || iterNum >= para.maxIter )
            converged = true;
            if iterNum >= para.maxIter
                disp('Maximum iterations reaches') ;
            end
        else
            prevObj = curObj;
        end
        
    end
end

timeConsumed = toc
disp(['total number of iterations: ' num2str(numIterInner) ]);
disp(['number of outer loop: '       num2str(numIterOuter) ]);

%% save the optimization results

Do = zeros(size(D));
for i = 1 : numImages
    Tfm = fliptform(maketform('projective', T_in{i}'));            
    y = vec(imtransform(I0{1,i}, Tfm, 'bicubic','XData', [1 imgSize(2)], 'YData', [1 imgSize(1)], 'Size', imgSize));
    y = y / norm(y) ;
    Do(:, i) = y;
end

if para.saveEnd
    save(fullfile(destDir, 'final.mat'), 'Do', 'A', 'E', 'tau');
end

outputFileName = fullfile(destDir, 'results.txt'); 
fid = fopen(outputFileName, 'a') ;
fprintf(fid, '%s\n', ['align images: '     num2str(numImages)    ]) ;
fprintf(fid, '%s\n', ['total iterations: ' num2str(numIterInner) ]) ;
fprintf(fid, '%s\n', ['outer loop: '       num2str(numIterOuter) ]) ;
fprintf(fid, '%s\n', ['elapsed time: '     num2str(timeConsumed) ]) ;
fprintf(fid, '%s\n', ['parameters:']) ;
fprintf(fid, '%s\n', ['   transformType: ' para.transformType ]) ;
fprintf(fid, '%s\n', ['   lambda: ' num2str(para.lambdac) ' / sqrt(' num2str(para.imageSize(1)) ')']) ;
fprintf(fid, '%s\n', ['   stop condition of outer loop: ' num2str(para.outer_tolerance) ]) ;
fprintf(fid, '%s\n', ['   stop condition of inner loop: ' num2str(para.inner_tolerance) ]) ;
fprintf(fid, '--------------------------------\n') ;
fclose(fid);
