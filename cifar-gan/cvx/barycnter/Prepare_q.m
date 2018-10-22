function [ q ] = Prepare_q( lambda,w,rho,D,K,mk )
%UNTITLED2 此处显示有关此函数的摘要
%   lambda lagrangians multiplier
%   w  weight
%   D  cost matrix

q = cell(K,1);
for i = 1:K
    q{i,1} = (lambda(i)-w(i))*ones(mk,1);
end
q = 2*rho*cell2mat(q)+reshape(D',[K*mk,1]);
end

