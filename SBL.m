%% EE798L: Machine Learning for Wireless Communications
% MATLAB Assignment-2: Sparse Bayesian Learning
% NAME: S.Srikanth Reddy; Roll No: 22104092

clear variables;
clc;

N=20;M=40;D_o=7;

variances_dB=[-20 -15 -10 -5 0]; % dB
variances = 10.^(variances_dB/10); 
B = 1./variances; % assume noise variance to be known, hence will not be updated

phi = normrnd(0,1,[N,M]); % generating design matrix

w = sparse(sort(randsample(1:40,D_o)),ones(1,D_o),normrnd(0,1,[1,D_o]),M,1); % generating sparse weight vector

w_hat = zeros(M,length(variances));
stop_constant = 10^-3; % will be used in stopping criteria in while loop
NMSE = zeros(1,length(variances));
for v=1:length(variances)
    epsilon = normrnd(0,sqrt(variances(v)),[N,1]); % generating noise entries
    
    t = phi*w + epsilon; % generating t
    
    mean_vec(:,1) = zeros(M,1); % initial mean vector with all zeros
    alpha = 100*ones(1,M); % alpha(i) = 100 ∀ i
    A = diag(alpha);
    cov_mat = inv(B(v)*(phi'*phi) + A); % equ (12) from R1
    s=2;
    mean_vec(:,s) = B(v)*cov_mat*phi'*t; % equ (13) from R1
    
    while (norm(mean_vec(:,s)-mean_vec(:,s-1))^2 > stop_constant*norm(mean_vec(:,s-1))^2)
        gamma = zeros(1,M);
        for j = 1:M
            gamma(j) = alpha(j)*cov_mat(j,j); % calculating gamma
            alpha(j) = (1-gamma(j))/mean_vec(j,s)^2; % updaing alpha
        end
        A = diag(alpha);
        s=s+1;
        cov_mat = inv(B(v)*(phi'*phi) + A); % calculating covariance matrix
        mean_vec(:,s) = B(v)*cov_mat*phi'*t; % calculating mean vector
    end
    w_hat(:,v) = mean_vec(:,s); % storing the estimate to be able to make predictions with new data
    NMSE(v) = (norm(w_hat(:,v)-w)^2)/(norm(w)^2); % calculating NMSE
end
plot(variances_dB,NMSE);
grid on;
xlabel('noise variance in dB');
ylabel('NMSE')
title('Normalized Mean Square Error plot')