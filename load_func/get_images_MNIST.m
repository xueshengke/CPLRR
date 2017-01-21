function [trainImages, testImages, trainTransform, testTransform] = get_images_MNIST( ...
    imagePath, para, transformInitType)
%--------------------------------------------------------------------------
% Xue Shengke, Zhejiang University, December 2016.
% Contact information: see readme.txt
%
% Partially composed of Peng Y., et al. RASL implementation, November 2009.
%--------------------------------------------------------------------------
%     load images of MNIST datasets
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
load(fullfile(imagePath, 'mnist_uint8.mat'));
trainNum = size(train_x, 1);
testNum = size(test_x, 1);
imgSize  = para.imageSize;
classNum = para.classNum;

%% resize and normalize images
train_x = double(train_x) / 255;
train_y = double(train_y);
test_x = double(test_x) / 255;
test_y = double(test_y);

train_x = reshape(train_x, [trainNum imgSize(1) imgSize(2)]);
train_x = permute(train_x, [3 2 1]);
train_y = train_y';
[train_y, ~] = find(train_y == 1);
train_y = train_y';

test_x = reshape(test_x, [testNum imgSize(1) imgSize(2)]);
test_x = permute(test_x, [3 2 1]);
test_y = test_y';
[test_y, ~] = find(test_y == 1);
test_y = test_y';

%% images grouped to categories
trainImages = cell(1, classNum);
for i = 1 : classNum
    temp = train_x(:, :, train_y==i);
    randnum = randperm(size(temp, 3));
    trainImages{i} = temp(:, :, randnum(1:trainBatchNum));
end

testImages = cell(1, classNum);
for i = 1 : classNum
    temp = test_x(:, :, test_y==i);
    randnum = randperm(size(temp, 3));
    testImages{i} = temp(:, :, randnum(1:testBatchNum));
end

%% add corruptions on images
if ratio > 0
% add max value noises    
for i = 1 : classNum
    randnum = rand(imgSize(1), imgSize(2), trainBatchNum);
    corrupt = randnum < ratio;
    maxVal = max(trainImages{i}(:));
    trainImages{i}(corrupt) = maxVal;
end

for i = 1 : classNum
    randnum = rand(imgSize(1), imgSize(2), testBatchNum);
    corrupt = randnum < ratio;
    maxVal = max(testImages{i}(:));
    testImages{i}(corrupt) = maxVal;
end

% add random value noises
% for i = 1 : classNum
%     randnum = rand(imgSize(1), imgSize(2), trainBatchNum);
%     corrupt = randnum < ratio;
%     points = sum(corrupt(:)==1);
%     maxVal = max(trainImages{i}(:));
%     trainImages{i}(corrupt) = maxVal * rand(points, 1);
% end
% 
% for i = 1 : classNum
%     randnum = rand(imgSize(1), imgSize(2), testBatchNum);
%     corrupt = randnum < ratio;
%     points = sum(corrupt(:)==1);
%     maxVal = max(testImages{i}(:));
%     testImages{i}(corrupt) = maxVal * rand(points, 1);
% end

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
