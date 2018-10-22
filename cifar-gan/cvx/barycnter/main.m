%% 生成二维高斯分布
clc;
clear;
N = 5;
NumOfSamples = 40;
K = 10;
mu = cell(1,N);
covariance = cell(1,N);
dense = 0;

SamplesInitial;

%% 生成样本
while 1
    samples = GenerateSamples(N,NumOfSamples,mu,covariance);
    [emp_distribution,Samples,mk] = SamplesToDistribution(samples);
    if mk == N * NumOfSamples
        break;
    end
end
% samples = GenerateSamples(N,NumOfSamples,mu,covariance);
% [emp_distribution,Samples,mk] = SamplesToDistribution(samples);

%% 初始化联合分布
PI = rand(K,mk);
PI = PI ./ sum(sum(PI)) ;
%% 初始化w,x
w = rand(K,1);
w = w / sum(w);
% lagrangians multiplier
lambda = zeros(1,K);
% prepare block matrix
diag_matrix = repmat({ones(mk)},1,K);
vecsize = K*mk;
%% 联合分布矩阵向量化，按行排列
JointDistribution = reshape(PI',[vecsize,1]);
%% rho
rho0 = 50;
nIter = 10;  % 外循环
Tadmm = 12;  % 内循环
gamma0 = 0.1:0.05:2 ;
Iter = size(gamma0,2);
% 运行结果保存
REC_PI = cell(1,Iter);
REC_w  = cell(1,Iter);
REC_x  = cell(1,Iter);
REC_wd = zeros(1,Iter);
REC_wd_update= zeros(1,Iter-1);
% 初始化x
Initial_x = rand(K,2);
x = zeros(K,2);
theta = 0;
tic;

loop = 1;
cnt = 1;
for gamma = gamma0
    
    for i = 1:nIter
        % with fixed w and PI,update x
        % x = theta * Initial_x + (1-theta) * UpdateX(PI,w,Samples,K);
        for idx = 1:K
            if w(idx) ~= 0
                xtmp = sum(repmat(PI(idx,:)',[1,2]).*Samples)/w(idx);
                x(idx,:) = theta * Initial_x(idx,:) + (1-theta) * xtmp;
            else
                x(idx,:) = Initial_x(idx,:);
            end
        end
        % record x for sparse weight
        Initial_x = x;
        % update cost matrix  K*mk
        D = CostMatrix(x,Samples,K,mk);
        % update rho
        % rho = rho0*sum(sum(D))/(vecsize);
        
        %% 内循环求解联合分布
        for j = 1:Tadmm
            % update w
            %[w,tmp] = update_w(JointDistribution,K,mk,lambda);
            [ w,tmp ] = SparseUpdateW( JointDistribution,mk,K,lambda,gamma,rho0 );
            % iterative update joint_distribution
            H = 2*rho0*blkdiag(diag_matrix{:,:});
            q = Prepare_q( lambda,w,rho0,D,K,mk );
            Aeq = repmat(eye(mk),[1,K]); % mk*mk*K
            beq = emp_distribution;       % mk*1
            init_J = JointDistribution;
            [JointDistribution] = quadprog(H, q, [], [], Aeq, beq, zeros(vecsize,1), []);
            % update lambda
            lambda = lambda + tmp -w;
        end
        
        PI = reshape(JointDistribution,[mk,K])';  % K*mk
        
    end

    REC_PI{1,loop} = PI;
    REC_w{1,loop} = w;
    REC_wd(loop) = sum(sum(PI.*D));
    REC_x{1,loop} = x;
    loop = loop + 1;
    if loop > 2
        REC_wd_update(cnt) = REC_wd(loop-1)-REC_wd(loop-2);
        cnt = cnt + 1;
    end
end
toc;
disp(['运行时间: ',num2str(toc)]);

K_means = find(REC_wd_update == max(REC_wd_update));
Plot;

















