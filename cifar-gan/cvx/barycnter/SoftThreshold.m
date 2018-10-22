function [ s ] = SoftThreshold( k,a )
%UNTITLED3 此处显示有关此函数的摘要
%   soft thresholding operator

if a > k
    s = a - k;
elseif abs(a) < k
    s = 0;
else
    s = a + k;
end

end

