
% for i = 1:Iter
%     figure();
%     SamplePlot(samples,N);
%     Xplot(REC_x{1,i},K,REC_w{1,i} );
% %     figure();
% %     JointDistributionPlot(REC_PI{1,i})
% end

% figure();
% stem(gamma0, REC_wd,'b+');
% figure();
% stem(gamma0(2:end), REC_wd_update,'r+');
% title('wd¸üÐÂ')
% 
K_Num = zeros(1,Iter);
for i = 1:Iter
    K_Num(i) = sum(REC_w{1,i}~=0); 
end

% figure();
% stem(gamma0,K_Num);

K_wd = zeros(Iter,2);
K_wd(:,1) = K_Num';
K_wd(:,2) = REC_wd';
K_wd