function [ ] = Xplot( x,K,w )
%UNTITLED6 此处显示有关此函数的摘要
%   绘制聚类中心
    for i = 1:K
        if w(i) ~=0
            plot(x(i,1),x(i,2),'b+');
            hold on;
        end
    end   
end

