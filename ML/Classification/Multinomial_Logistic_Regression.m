% Multinomial Logistic Regression

load iris_dataset.mat;

% It is also possible to handle multiple classes. In the case we want to
% discriminate among three different iris classes we might do:

x = zscore(irisInputs([1 2],:)');

[t, ~] = find(irisTargets ~= 0);
[B_mul, dev_mul, stats_mul] = mnrfit(x,t);

phi_hat = mnrval(B_mul,x);
[~, t_pred] = max(phi_hat,[],2);
confusionmat(t,t_pred)

figure();
gscatter(x(:,1),x(:,2),t);
hold on;
axis manual

[a, b] = meshgrid(-3:0.1:3,-3:0.1:4);
axis tight
phi_hat = mnrval(B_mul,[a(:),b(:)]);
[~, pred] = max(phi_hat,[],2);
gscatter(a(:),b(:),pred);
