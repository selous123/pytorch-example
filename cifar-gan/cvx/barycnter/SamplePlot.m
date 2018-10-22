function [ ] = SamplePlot( samples,N )
%UNTITLED4 此处显示有关此函数的摘要

%   绘制样本点分布图
%   将N个分布的样本点绘制在一张图中

    for i = 1:N
        plot(samples{i}(:,1),samples{i}(:,2),'r+');
        hold on;
    end

end

