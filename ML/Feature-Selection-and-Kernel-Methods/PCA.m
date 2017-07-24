% PCA

clear
clc
close all

% What to do in the case the model you are considering is not performing
% well even by tuning properly the parameters (cross-validation) to
% discriminate between classes. We have two opposite options:
% 
% 1. Reduce the dimensionality of the data since most of them do not bring
% any information to the taks, thus they only introduce noise in the
% estimation procedure.
%
% 2. Consider high dimensionality where the classification task is easy, 
% by enhancing the input space with new features.
% 
% PCA is a technique to perform dimensionality reduction, i.e., to extract
% some low dimensional features from a dataset. More specifically, we find
% a linear transformation of the original data X s.t. the greatest variance
% lies in the first coordinate, the second greatest variance on the second
% coordinate ... It results that the new coordinates are orthogonal to each
% other.

% A procedure to compute these coordinates is the following:
%
% 1. Translate the original data X to X_HAT s.t. they have zero mean.
% 2. Compute the covariance matrix of X_HAT, C = X_HAT^T * X_HAT
% 3. The eigenvector e_1 corresponding to the largest eigenvalue lambda_1
% is the first principal component and so on
% 4. etc

load iris_dataset;
[irisTargets, ~] = find(irisTargets == 1);
gplotmatrix(irisInputs',[],irisTargets);
irisInputs = irisInputs';

% In this case we do not noramlize the data since otherwise we are loosing
% information on the directions with largest variance. Still we require
% that the variables have comparable ranges (e.g. the order of magnitude of
% all the variables is the same).

[loadings, scores, variance] = pca(irisInputs);

% loadings is the matrix identifying the linear transformation of the data W.
% Column vectors are the eigenvectors of the covariance matrix. scores are
% the transformed version of the original dataset (T = X_HAT * W) and
% variance corresponding to each one of the princial components we
% computed.

cumsum(variance) / sum(variance)

pc1 = scores(:,1);
pc2 = scores(:,2);
pc3 = scores(:,3);
pc4 = scores(:,4);

% Until this point we simply transform the original dataset, but we also
% need to reduce its dimension. If we consider only a subset of the columns
% of the loadings w, we effectively have a new representation of the
% original samples. Some way of choosing the number of principal components
% to consider are:

% - Keep all the principal components until we have a cumulated variance of
% 90 - 95%
% - Keep all the principal components which have more than 5% of the
% variance
% - Find the elbow in the cumulated variance

% There are multiple purposes to perform the PCA and project the dataset
% into a lower dimensional space. At first, we could consider the PCs as
% feature extraction technique for classification and regression tasks. Let
% us consider a logistic regression trained over a different set of data,
% e.g the original one, the first two PCs and the last two.

irisT = [ones(50,1); 2*ones(100,1)];
model_all = mnrfit(irisInputs, irisT);
model_pca = mnrfit([pc1, pc2], irisT);
model_acp = mnrfit([pc3, pc4], irisT);

prob_all = mnrval(model_all, irisInputs);
class_all = 1 * (prob_all(:,1) > prob_all(:,2)) + 2 * (prob_all(:,1) <= prob_all(:,2));
sum(class_all == irisT)

prob_pca = mnrval(model_all, irisInputs);
class_pca = 1 * (prob_pca(:,1) > prob_pca(:,2)) + 2 * (prob_pca(:,1) <= prob_pca(:,2));
sum(class_pca == irisT)

prob_acp = mnrval(model_acp, irisInputs);
class_acp = 1 * (prob_acp(:,1) > prob_acp(:,2)) + 2 * (prob_acp(:,1) <= prob_acp(:,2));
sum(class_acp == irisT)

% As you can see the use of the first two principal components gives
% similar results when we are performing a classification task with an SVM,
% since they are considering 99% of the total variance of the original
% data. As a remark we should underline that the PCA does not make use of
% the labels when it is performed on a dataset, while other feature
% selection techniques are based on the performance of the classifier used
% a posteriori of the feature selection phase. 

% PCA can also be used as a visualization tool. In fact, if we focus only
% on the first two / three PCs, we are able to plot datasets which are in
% principle in high dimensional spaces.

figure();
gscatter(pc1,pc2,irisTargets,'bg','..')
figure();
gscatter(pc3,pc4,irisTargets,'bg','..')

% At last, PCA can be used as a compression tool. In fact, the considered 
% linear transformation W minimizes among the ones with a given dimension.
% i.e., is the linear transformation minimizing the reconstruction error. 
% Let visualize this effect by plotting the distances between original and 
% reconstructed points.

m = 2

x_zip = scores(:,1:m);
needed_loadings = loadings(:,1:m);
mean_values = mean(irisInputs);

x_rec = x_zip * needed_loadings' + repmat(mean_values,150,1);

mean((irisInputs - x_rec).^2)

figure();
xlim([4.3 7.9]);
ylim([2 4.4]);
for ii = 1:150
    hold on;
    plot([irisInputs(ii,1) x_rec(ii,1)],[irisInputs(ii,2) x_rec(ii,2)],'k');
end

% The more the principal component we consider the less the reconstruction 
% error. We will have a perfect reconstruction if we consider all the 
% principal components.