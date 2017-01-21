function [trainImages, testImages, trainTransform, testTransform] = get_images_COIL( ...
    imagePath, para, transformInitType)
%--------------------------------------------------------------------------
% Xue Shengke, Zhejiang University, December 2016.
% Contact information: see readme.txt
%
% Partially composed of Peng Y., et al. RASL implementation, November 2009.
%--------------------------------------------------------------------------
%     load images of COIL datasets
% 
%     Inputs:
%         imagePath          -- path where image data exists
%         para               -- patameters for loading the dataset
%         transformInitType  -- initialization transformation type
% 
%     Outputs:
%         trainImages        -- train images categories
%         testImages         -- test images categories
%         trainTransform     -- transformations for train images
%         testTransform      -- transformations for test images
%--------------------------------------------------------------------------
%% read parameters from para
baseCoords    = para.coordinate;
trainBatchNum = para.trainImageNum;
testBatchNum  = para.testImageNum;
ratio         = para.noise_ratio;

%% load original data from *.mat file
load(fullfile(imagePath, 'COIL-20.mat'), 'coilData');
imgSize  = para.imageSize;
classNum = para.classNum;

%% resize and normalize images
normCOIL = cell(classNum, 1);
for i = 1 : classNum
    tmp = coilData{i} / 255;
    normCOIL{i} = imresize(tmp, [imgSize(1) imgSize(2)]);
end

%% images grouped to categories
trainImages = cell(classNum, 1);
testImages  = cell(classNum, 1);
for i = 1 : classNum
    images = normCOIL{i};
    imageNum = size(images, 3);
    if imageNum < trainBatchNum + testBatchNum 
        fprintf('ERROR: class %d, images are not enough, %d < %d + %d !\n', ...
            i, imageNum, trainBatchNum, testBatchNum);
        return ; 
    end
    randnum = randperm(imageNum);    
    trainImages{i} = images(:, :, randnum(1:trainBatchNum));
    testImages{i} = images(:, :, randnum(trainBatchNum+1 : trainBatchNum+testBatchNum));
end

%% add corruptions on images
if ratio > 0
% add max value noises
% for i = 1 : classNum
%     randnum = rand(imgSize(1), imgSize(2), trainBatchNum);
%     corrupt = randnum < ratio;
%     maxVal = max(trainImages{i}(:));
%     trainImages{i}(corrupt) = maxVal;
% end
% 
% for i = 1 : classNum
%     randnum = rand(imgSize(1), imgSize(2), testBatchNum);
%     corrupt = randnum < ratio;
%     maxVal = max(testImages{i}(:));
%     testImages{i}(corrupt) = maxVal;
% end

% add random value noises
for i = 1 : classNum
    randnum = rand(imgSize(1), imgSize(2), trainBatchNum);
    corrupt = randnum < ratio;
    points = sum(corrupt(:)==1);
    maxVal = max(trainImages{i}(:));
    trainImages{i}(corrupt) = maxVal * rand(points, 1);
end

for i = 1 : classNum
    randnum = rand(imgSize(1), imgSize(2), testBatchNum);
    corrupt = randnum < ratio;
    points = sum(corrupt(:)==1);
    maxVal = max(testImages{i}(:));
    testImages{i}(corrupt) = maxVal * rand(points, 1);
end

end
%% create transformations for each image
trainTransform = cell(trainBatchNum, 1);
testTransform = cell(testBatchNum*classNum, 1);

if strcmp(transformInitType,'IDENTITY')
    for index = 1 : trainBatchNum
        trainTransform{index} = eye(3) ;
    end
    for index = 1 : testBatchNum*classNum
        testTransform{index} = eye(3) ;
    end
elseif strcmp(transformInitType,'SIMILARITY')
    for index = 1 : trainBatchNum
        points = baseCoords;
        S = TwoPointSimilarity( baseCoords, points );
        trainTransform{index} = S ;
    end
    for index = 1 : testBatchNum*classNum
        points = baseCoords;
        S = TwoPointSimilarity( baseCoords, points );
        testTransform{index} = S ;
    end    
elseif strcmp(transformInitType,'AFFINE')
    for index = 1 : trainBatchNum
        points = [baseCoords, [0,1]'];
        S = ThreePointAffine( baseCoords, points );
        trainTransform{index} = S ;
    end
    for index = 1 : testBatchNum*classNum
        points = [baseCoords, [0,1]'];
        S = ThreePointAffine( baseCoords, points );
        testTransform{index} = S ;
    end    
else
    error('unable to initialize the transformations!');
end
