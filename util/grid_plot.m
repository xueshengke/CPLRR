function [ ] = grid_plot( D, numImage, imageSize, layout )
%--------------------------------------------------------------------------
% Xue Shengke, Zhejiang University, December 2016.
% Contact information: see readme.txt
%
% Partially composed of Peng Y., et al. RASL implementation, November 2009.
%--------------------------------------------------------------------------
%   display image in D, each column is a stacked vector of a 2-D image
%   
%   Inputs:
%       D             --- dictionary for displaying each column
%       numImage      --- number of images to display
%       imageSize     --- image size
%       layout        --- layout to display images
%
%   Outputs: 
%--------------------------------------------------------------------------
% layout
if nargin < 4
    xI = ceil(sqrt(numImage)) ;
    yI = ceil(numImage/xI) ;
else
    xI = layout.xI;
    yI = layout.yI;
end
gap = 2;
gap2 = 1;
container = ones(imageSize(1)+gap, imageSize(2)+gap); 
% white edges
bigpic = cell(xI,yI); % (xI*canonicalImageSize(1),yI*canonicalImageSize(2));

for i = 1:xI
    for j = 1:yI
        if yI*(i-1)+j > numImage
            bigpic{i,j} = ones(imageSize(1)+gap, imageSize(2)+gap);
        else
            container((1+gap2):(end-gap2), (1+gap2):(end-gap2)) = reshape(D(:,yI*(i-1)+j), imageSize);
            bigpic{i,j} = container;
        end
    end
end

imshow(cell2mat(bigpic),[],'DisplayRange',[0 max(max(D))],'Border','tight')
end