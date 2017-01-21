function [ out ] = indicator( i, j, beta )
%INDICATOR Summary of this function goes here
%   Detailed explanation goes here
if i == j
    out = beta;
else
    out = -1;
end

end

