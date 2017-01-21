function [ accuracy, overall ] = projection_predict( testImages, ...
    transformations, para, destDir )
%--------------------------------------------------------------------------
% Xue Shengke, Zhejiang University, December 2016.
% Contact information: see readme.txt
%
%--------------------------------------------------------------------------
%   predict labels for test images
%   
%   Inputs:
%       testImages        --- test images
%       transformations   --- initial transformation
%       para              --- parameters struct
%       destDir           --- save directory
%
%   Outputs: 
%       accuracy          --- predict accuracy of each category
%       overall           --- predict accuracy of entire images
%--------------------------------------------------------------------------

classNum = para.classNum;
imgSize = para.imageSize;
dim = length(size(testImages{1}));

testData = [];
testLabel = [];
predicts = zeros(size(testLabel));
accuracy = zeros(classNum, 1);
labelNum = zeros(classNum, 1);

%% construct test data and labels
for i = 1 : classNum
    imageNum = size(testImages{i}, dim);
    labelNum(i) = imageNum;
    testData = cat(dim, testData, testImages{i});
    lab = i * ones(1, imageNum);
    testLabel = [testLabel lab];
end

%% check if colorful images
totalNum = size(testData, dim);
if dim == 4
    tmpData = zeros(imgSize(1), imgsize(2), totalNum);
    for i = 1 : totalNum
        tmpData(:, :, i) = rgb2gray(testData(:, :, :, i));
    end
    testData = tmpData;
end

%% load projection parameters
W = cell(classNum, 1);
B = cell(classNum, 1);
A = cell(classNum, 1);
for i = 1 : classNum
    strc = load(fullfile(destDir{i}, 'projection.mat'), 'W', 'B');
    W{i} = strc.W;
%     B{i} = strc.B;
%     B{i} = mean(B{i}, 2);
%     B{i} = B{i} / max(B{i});
    strc = load(fullfile(destDir{i}, 'final.mat'), 'A');
    Am = mean(strc.A, 2);
    A{i} = Am / max(Am);    
end

%% predict a batch of test images, every 100 images
actv = para.actv_func;
batch = 100;
for i = 1 : ceil(totalNum / 100)
    fprintf('%d / %d \n', i*100, totalNum);
    batchImages = testData(:, :, 1+(i-1)*batch : min(i*batch,totalNum));
    numImages = size(batchImages, 3); 
    batchImages = reshape(batchImages, [prod(imgSize) numImages]);
    plabel = zeros(classNum, numImages);  

    for k = 1 : classNum
        Z = actv(W{k} * batchImages);
        minZ = repmat(min(Z), [size(Z, 1) 1]);
        maxZ = repmat(max(Z), [size(Z, 1) 1]);
        Z = (Z - minZ) ./ (maxZ - minZ);
        err = repmat(A{k}, [1 numImages]) - Z;
        plabel(k, :) = sum(abs(err));
        
        if para.DISPLAY
            figure;
            xI = ceil(sqrt(classNum));
            yI = ceil(classNum / xI);
            subplot(2*xI, yI, k);
            grid_plot(Z, size(Z,2), [imgSize(1) imgSize(2)]); title('activation');
            subplot(2*xI, yI, k+xI*yI); 
            grid_plot(err, size(err,2), [imgSize(1) imgSize(2)]); title('error');
        end
    end

    [~, predID] = min(plabel);
    predicts(1+(i-1)*batch : min(i*batch,totalNum)) = predID;   
end

%% calculate predict accuracy
overall = sum(testLabel == predicts) / totalNum;
[~, index] = find(testLabel == predicts);
identity = testLabel(index);
for k = 1 : classNum
    accuracy(k) = sum(identity == k) / labelNum(k);
end

end
