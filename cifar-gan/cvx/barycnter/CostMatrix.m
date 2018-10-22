function [ D ] = CostMatrix( x,Samples,K,mk )
%UNTITLED 此处显示有关此函数的摘要
%   计算中心到样本的距离矩阵
%   L2
D = zeros(K,mk);
for i = 1:K
    for j = 1:mk
        D(i,j) = norm(x(i,:)-Samples(j,:),2);
    end
end
end

