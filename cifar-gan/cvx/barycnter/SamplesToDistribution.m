function [ empirical_distribution,uniqueSamples,uniqueSize ] = SamplesToDistribution( samples )
%SAMPLESTODISTRIBUTION 此处显示有关此函数的摘要
%   根据所有的样本统计经验分布
%   samples (1,N) NumOfSamples * 2

% support point
N = size(samples,2);  %% 分布的个数
NumOfSamples = size(samples{1,1},1);  %% 每个分布的个数

X = zeros(N*NumOfSamples,2);  % 矩阵的形式存储所有分布的样本点
for i = 1:N
    for j = 1:NumOfSamples
        X((i-1)*NumOfSamples+j,:)=samples{i}(j,:);
    end    
end

%精度约束
% X(:,:) = round(X(:,:));
uniqueSamples = unique(X,'rows');
uniqueSize = size(uniqueSamples,1);
Count =zeros(1,uniqueSize);
for i = 1 : uniqueSize
    for j = 1 : N*NumOfSamples
        if uniqueSamples(i,:)==X(j,:)
            Count(1,i)=Count(1,i)+1;
        end
    end
end


empirical_distribution = (Count / sum(Count))';

end

