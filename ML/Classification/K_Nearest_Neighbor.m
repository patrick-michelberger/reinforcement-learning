% K-nearest neighbor

% Finally, we classify the iris dataset by considering a non-parametric
% method, for instance a K-NN classifier.
%
% - A distance metric: Euclidean
% - How many neighbours: k = 3
% - A weight function: no weights
% - How to fit with local points: Majority vote (break ties with lowest
% index class)

load iris_dataset.mat;
x = zscore(irisInputs([1 2],:)');
t = irisTargets(1,:)';

knn_model = fitcknn(x, t, 'NumNeighbors', 3);
t_pred = predict(knn_model,x);
confusionmat(t, t_pred)

figure();
gscatter(x(:,1),x(:,2),t);
hold on;
axis manual

[a, b] = meshgrid(-3:0.1:3,-3:0.1:4);
axis tight
pred = predict(knn_model,[a(:),b(:)]);
gscatter(a(:),b(:),pred);
title('K-NN classifier');

% Which does not require any learning phase! If we change K we can notice
% how the decision boundaries come smoother or rougher.