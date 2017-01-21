function RASL_plot(destDir, numImage, imageSize, layout)
%--------------------------------------------------------------------------
% Xue Shengke, Zhejiang University, December 2016.
% Contact information: see readme.txt
%
% Partially composed of Peng Y., et al. RASL implementation, November 2009.
%--------------------------------------------------------------------------
%   display image results after RASL
%   
%   Inputs:
%       destDir           --- directory in which stores results of RASL
%       numImage          --- number of images to display
%       imageSize         --- image size
%       layout            --- layout to display images
%
%   Outputs: 
%--------------------------------------------------------------------------
%% load data
% initial input images
load(fullfile(destDir, 'original.mat'), 'D') ;

% alignment results
load(fullfile(destDir, 'final.mat'), 'Do', 'A', 'E') ;

%% display

% layout
if nargin < 4
    xI = ceil(sqrt(numImage)) ;
    yI = ceil(numImage/xI) ;

    gap = 2;
    gap2 = 1; % gap2 = gap/2;
else
    xI = layout.xI ;
    yI = layout.yI ;

    gap = layout.gap ;
    gap2 = layout.gap2 ; % gap2 = gap/2;
end
container = ones(imageSize(1)+gap, imageSize(2)+gap); 
% white edges
bigpic = cell(xI,yI);

% D
for i = 1:xI
    for j = 1:yI
        if yI*(i-1)+j > numImage
            bigpic{i,j} = ones(imageSize(1)+gap, imageSize(2)+gap);
        else
            container((gap2+1):(end-gap2), (gap2+1):(end-gap2)) = reshape(D(:,yI*(i-1)+j), imageSize);
            bigpic{i,j} = container;
        end
    end
end
figure;
subplot(2,2,1);
imshow(cell2mat(bigpic),[],'DisplayRange',[0 max(max(D))],'Border','tight');
title('Input images');

% Do
for i = 1:xI
    for j = 1:yI
        if yI*(i-1)+j > numImage
            bigpic{i,j} = ones(imageSize(1)+gap, imageSize(2)+gap);
        else
            container((gap2+1):(end-gap2), (gap2+1):(end-gap2)) = reshape(Do(:,yI*(i-1)+j), imageSize);
            bigpic{i,j} = container;
        end
    end
end
% figure
subplot(2,2,2);
imshow(cell2mat(bigpic),[],'DisplayRange',[0 max(max(Do))],'Border','tight');
title('Aligned images');

% A
for i = 1:xI
    for j = 1:yI
        if yI*(i-1)+j > numImage
            bigpic{i,j} = ones(imageSize(1)+gap, imageSize(2)+gap);
        else
            container((gap2+1):(end-gap2), (gap2+1):(end-gap2)) = reshape(A(:,yI*(i-1)+j), imageSize);
            bigpic{i,j} = container;
        end
    end
end
% figure
subplot(2,2,3);
imshow(cell2mat(bigpic),[],'DisplayRange',[0 max(max(A))],'Border','tight');
title('Low-rank components');

% E
for i = 1:xI
    for j = 1:yI
        if yI*(i-1)+j > numImage
            bigpic{i,j} = ones(imageSize(1)+gap, imageSize(2)+gap);
        else
            container((gap2+1):(end-gap2), (gap2+1):(end-gap2)) = reshape(E(:,yI*(i-1)+j), imageSize);
            bigpic{i,j} = container;
        end
    end
end
% figure
subplot(2,2,4);
imshow(abs(cell2mat(bigpic)),[],'DisplayRange',[0 max(max(abs(E)))],'Border','tight');
title('Sparse corruptions');

figure;
subplot(2,2,1);
imshow(reshape(sum(D,2), imageSize), []);
title('average of misaligned D');
subplot(2,2,2);
imshow(reshape(sum(Do,2), imageSize), []);
title('average of aligned D');
subplot(2,2,3);
imshow(reshape(sum(A,2), imageSize), []);
title('average of A');
