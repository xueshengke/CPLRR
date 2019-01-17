% Xue Shengke, Zhejiang University, December 2016. 
% Contact information: see readme.txt
%
% Reference: 
% Shengke Xue, Xinyu Jin, "Robust Classwise and Projective Low-Rank Representation 
% for Image Classification", Signal, Image and Video Processing, 2018, 12(1):107-115. 
% DOI 10.1007/s11760-017-1136-1
% 
% Partially composed of Peng, Y., et al. RASL implementation, November 2009.
%% clear
clear;
close all;
clc;

%% addpath
addpath data ;
addpath load_func ;
addpath CPLR_func ;
addpath RASL_func ;
addpath result ;
addpath util ;

%% define data path

currentPath = cd ;

% input path
imagePath = fullfile(currentPath, 'data') ;
load('COIL-20.mat', 'className');
userName = className;
classNum = numel(userName);

% output path
destDir = cell(classNum, 1);
destRoot = fullfile(currentPath, 'result', 'COIL') ;
for i = 1 : classNum
    destDir{i} = fullfile(destRoot, userName{i}) ;   
    if ~exist(destDir{i}, 'dir'),   mkdir(destRoot, userName{i}); end
end

%% define parameters
trainMode = 0;
testMode = 0;

trainMode = 1;
testMode = 1;

% dispaly flag
para.DISPLAY = 0 ;
para.DISPLAY = 1 ;

% save flag
para.saveStart = 1 ;
para.saveEnd = 1 ;
para.saveIntermedia = 0 ;

% for windows images
% para.imageSize = [ 32  32 ];
% para.coordinate = [ 5  28 ; ...
%                    17  17 ];
para.imageSize = [ 128  128 ];
para.coordinate = [ 17  112 ; ...
                   65  65 ];
               
% parametric tranformation model
para.transformType = 'EUCLIDEAN'; 
% one of 'TRANSLATION', 'EUCLIDEAN', 'SIMILARITY', 'AFFINE','HOMOGRAPHY'

para.numScales = 2 ; % if numScales > 1, we use multiscales

% main loop in RASL
para.outer_tolerance = 1e-2; % stop iteration threshold
para.maxIter = 50;           % maximum iteration

% inner loop in RASL
para.inner_tolerance = 1e-7; % stop iteration threshold
para.inner_maxIter = 1000;   % maximum iteration
para.lambdac = 1 ;           % lambda = lambdac / sqrt(m) for ||E||_1

trainImageNum = 10;                 % train image number for each category
testImageNum  = 72 - trainImageNum; % test image number for each category
para.trainImageNum = trainImageNum;
para.testImageNum  = testImageNum;
para.classNum = classNum;

para.noise_ratio = 0.0;   % 0, 0.1, 0.2, 0.3
para.alpha = 0.9;         % momentum
para.gamma = 3;           % 5, 15, 25, 35
para.beta = 20;           % equal to the number of class 
para.lr = [1e-2, 1e-3, 1e-4, 1e-5];
para.proj_threshold = 1e-4;
para.proj_maxIter = 1000;
para.actv_func = @sigmoid;       
para.der_actv_func = @der_sigmoid; 

%% Get training images
transformationInit = 'SIMILARITY';
[trainImages, testImages, trainTransform, testTransform] = get_images_COIL(imagePath, para, transformationInit);

%% start RASL training
raslTimeElapsed = 0;
if trainMode
tic
for i = 1 : classNum
    disp( '--------------');
    disp(['begin RASL training ' userName{i}]);

    % RASL, alignment batch images that belongs to the same category
    [Di, Do, A, E, xi, numIterOuter, numIterInner] = RASL_main(trainImages{i}, trainTransform, para, destDir{i});
    
    % plot the results
    if para.DISPLAY
        layout.xI = ceil(sqrt(trainImageNum)) ;
        layout.yI = ceil(trainImageNum / layout.xI) ;
        layout.gap = 2 ;
        layout.gap2 = 1 ;
        RASL_plot(destDir{i}, para.trainImageNum, para.imageSize, layout);
    end     
end
fprintf('RASL training completes!\n');
raslTimeElapsed = toc
end

%% start projection function training
projTimeElapsed = 0;
if trainMode
tic
D = cell(classNum, 1);
A = cell(classNum, 1);
disp( '--------------');
disp('load data D and A from ');
for i = 1 : classNum
    fprintf('%s, ' ,userName{i});
    D{i} = load(fullfile(destDir{i}, 'original.mat'), 'D');
    A{i} = load(fullfile(destDir{i}, 'final.mat'), 'A');
end
disp(' ');

% learn the projection function class-wise
W = cell(classNum, 1);
B = cell(classNum, 1);
for i = 1 : classNum
    disp( '--------------');
    disp(['begin Projection training ' userName{i}]);
    [W{i}, B{i}] = projection_train(D, A, i, para, destDir{i});
end
fprintf('Projection training completes!\n');
projTimeElapsed = toc
end

%% start test
testTimeElapsed = 0;
if testMode
fprintf('begin test ---------------\n');
tic
[accuracy, overall] = projection_predict(testImages, testTransform, para, destDir);
fprintf('classification result: %.4f\n', accuracy);
fprintf('overall accuracy: %.4f\n', overall);
fprintf('test completes!\n');
testTimeElapsed = toc

%% record test results
outputFileName = fullfile(destRoot, 'COIL_parameters.txt'); 
fid = fopen(outputFileName, 'a') ;
fprintf(fid, '****** %s ******\n', datestr(now,0));
fprintf(fid, '%s\n', ['image size: '  num2str(para.imageSize(1)) ' x ' num2str(para.imageSize(2)) ]);
fprintf(fid, '%s\n', ['image class: '     num2str(para.classNum)     ]);
fprintf(fid, '%s\n', ['train number: '    num2str(para.trainImageNum)]);
fprintf(fid, '%s\n', ['test number: '     num2str(para.testImageNum) ]);
fprintf(fid, '%s\n', ['RASL time: '       num2str(raslTimeElapsed)   ]);
fprintf(fid, '%s\n', ['Projection time: ' num2str(projTimeElapsed)   ]);
fprintf(fid, '%s\n', ['test time: '       num2str(testTimeElapsed)   ]);
fprintf(fid, 'parameters:\n');
fprintf(fid, '%s\n', ['  outer stop: '    num2str(para.outer_tolerance)]);
fprintf(fid, '%s\n', ['  outer maxIter: ' num2str(para.maxIter)        ]);
fprintf(fid, '%s\n', ['  inner stop: '    num2str(para.inner_tolerance)]);
fprintf(fid, '%s\n', ['  inner maxIter: ' num2str(para.inner_maxIter)  ]);
fprintf(fid, '%s\n', ['  lambdac: '       num2str(para.lambdac)        ]);
fprintf(fid, '%s\n', ['  corrupt: '       num2str(para.noise_ratio)    ]);
fprintf(fid, '%s\n', ['  alpha: '         num2str(para.alpha)          ]);
fprintf(fid, '%s\n', ['  gamma: '         num2str(para.gamma)          ]);
fprintf(fid, '%s\n', ['  nu: '            num2str(para.beta)           ]);
fprintf(fid, '  lr: ');
fprintf(fid, '%s, ', num2str(para.lr(1)) );
fprintf(fid, '%s, ', num2str(para.lr(2)) );
fprintf(fid, '%s, ', num2str(para.lr(3)) );
fprintf(fid, '%s\n', num2str(para.lr(4)) );
fprintf(fid, '%s\n', ['  projTol: '     num2str(para.proj_threshold)]);
fprintf(fid, '%s\n', ['  projMaxIter: ' num2str(para.proj_maxIter)  ]);
fprintf(fid, '%s\n', '  activation: sigmoid');

for i = 1 : length(accuracy)
    fprintf(fid, '%s\n', ['class: ' num2str(i) ', accuracy: ' num2str(accuracy(i))]);
end
fprintf(fid, '%s\n', ['overall accuracy: ' num2str(overall)]);
fprintf(fid, '--------------------\n');
fclose(fid);
end
