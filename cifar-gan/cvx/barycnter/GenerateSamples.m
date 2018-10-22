function [ output_Samples ] = GenerateSamples( N,NumOfSamples,mu,covariance )
%GENERATESAMPLES 此处显示有关此函数的摘要
%  N 二维高斯分布个数
%  NumOfSamples 每个二维高斯分布的样本个数
%  mu  2*N 高斯分布均值矩阵
%  covariance 高斯分布协方差矩阵

if N~=size(mu,2)
    fprintf('高斯分布个数不匹配');
    return;
end
output_Samples = cell(1,N);
for i = 1:N
    output_Samples{1,i} = mvnrnd(mu{i},covariance{i},NumOfSamples);
end

