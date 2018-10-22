function [ X ] = UpdateX( PI,w,Samples,K )
%UNTITLED2 此处显示有关此函数的摘要
%   X 为样本对后验概率的加权和
%   X K*2
X = zeros(K,2);
for i = 1:K
    X(i,:) = sum(repmat(PI(i,:)',[1,2]).*Samples)/w(i); % PI(i,:)/w(i) 样本集对后验概率的加权和
end

end

