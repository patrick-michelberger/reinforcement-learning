% Logistic Regression

load iris_dataset.mat;
x = zscore(irisInputs([1 2],:)');
t = irisTargets(1,:)';
gplotmatrix(x, [], t);

% Let us perform a classification using Logistic Regression.
% 
% - Hypothesis space: y(x) = o(w_0 + w_1 * x_1 + w_2 * x_2)
% - Loss function: MLE L(w) = - SUM(C*ln(y) + (1 - C) * ln(1-y))
% - Optimization method: Gradient descent
%
% where o(x) = 1 / (1 + e^x)

t = t + 1;
[B, dev, stats] = mnrfit(x,t)

% We add one to the targets t since mnrfit considers categorical classes
% corresponding to positive integers. In the structure stats we have also
% information about the statistical significance of the learned parameters
% since if we perform the logit(x) = log(X/(1-x)) transformation to the
% output we have: 
% 
% logit(y) = w_0 + x_1 * w_1 + x_2 * w_2
%
% Thus we have the same statistical characterication of the parameters w as
% we had in the linear regression if we consider as output a specific
% transformation of the target.

phi_hat = mnrval(B,x);
[~, t_pred] = max(phi_hat,[],2);
confusionmat(t,t_pred)

figure();
gscatter(x(:,1),x(:,2),t);
hold on;
axis manual;

h(1) = plot(s, -(s * B(2) + B(1) ) / B(3),'b');
