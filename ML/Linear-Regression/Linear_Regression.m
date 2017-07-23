% Linear Regression

% Let us consider the iris_dataset. In the dataset we have data regarding
% specific species of flowers:
%
% - Sepal length
% - Sepal width
% - Petal length
% - Petal width
% - Species (Iris setosa, Iris virginica and Iris versicolor)
%
% in the specific, N = 150 total samples (50 per type)

% At first, we want to predict the petal width of a specific kind of Iris
% setosa by using the petal length. This can be considered a regression
% problem where we consider as feature x the petal length an as a target
% t the petal width. In order to provide a prediction t_hat for the target
% t we will consider 
% 
% - Hypothesis space: t_hat = f(x,w) = w_0 + x*w_1
% - Loss measure: J(w, x, t) = RSS(w) = SUM(t_hat - t)^2
% - Optimization method: Least Square (LS) method
%
% where w e R^M, M = 2

% 1. Data Preprocessing

load iris_dataset.mat;

% Before we start analyzing the data, one should plot the considered data
% to inspect it.

figure();
gplotmatrix(irisInputs');
x = irisInputs(3,:)';
t = irisInputs(4,:)';

% Once we have inspect the data, we should operate some pre-processing
% procedures. On a generic dataset one should perform:
%
% - shuffeling
% - remove inconsistent data
% - remove outliers
% - normalize or standardize data
% - fill missing data
% 
% We simply normalize the data by using the function zscore() which
% substracts the sample mean of the dataset and divides by the estimates of
% the standard deviation. 

x = zscore(x);
t = zscore(t);
figure();
plot(x,t, 'bo');

% 2. Linear Regression Options 

% Once we pre-processed the data, we solve the regression problem. In the
% recap of the figure of merits that allow us to understand if the fitting
% we considered was valuable:
%
% - Residual sum of squares (sse): How much does the prediction differ from
% the true value?
%
% - R^2 (rsquare): How the fraction of the variance of the data explained
% by the model ? 
%
% - Degrees of freedom (dfe): dfe = N - M How much is the model flexible in
% fitting the data?
%
% - adjusted R^2: (adjrsquare) rsquare corrected by how much flexibility the
% model has
%
% - Root Mean Square Error (RMSE): normalized version of RSS
%
% Moreover it is possible to have a confidence interval for the estimated
% coefficients w. In fact, it is possible to show that under the assumption
% that the observations t are i.i.d. and satisfies t = wX + eps where eps
% is a Gaussian noise with zero mean and variance (the data are generated
% by a linear model with noise), the computed coefficients w, are
% distributed as a t-distribution.

%fit_specifications = fittype( {'1', 'x'}, 'independent', 'x', 'dependent', 't', 'coefficients', {'w0', 'w1'} );
%[fitresult, gof] = fit( x, t, fit_specifications);

% The results of the fitting process are logged in the console. Also here
% we have information about the confidence intervals of the parameters and
% about common goddness of fit values, to evaluate if the chosen model is
% consistent with the real relationship existing between input and target.

% Similarly, we can use the function fitlm() which is more general to the
% previously ones, since it allow us to perform even multiple regression.

ls_model = fitlm(x,t)

% Finally, we can implement the function to perform LS fitting by scartch.

n_samples = length(x);
Phi = [ones(n_samples,1) x];
mpinv = pinv(Phi' * Phi) * Phi';
w = mpinv * t

% By using pinv() we are implicitly using the default tolerance for the
% eigenvalues of a matrix A which has value: max(size(A)) * norm(A) * eps

% Note: Here we considered the feature matrix Phi with a constant term and
% a linear term. This matrix can be expanded by adding new columns eg. if
% we want to consider also a quadratic term we write:

% Phi = [ones(n_samples,1) x x.^2];

% Clearly by using this last option we are more aware of the tools we are
% using and we are faster in the computation of the solution, but we have
% compute the figures of merit by hand.

hat_t = Phi * w;
bar_t = mean(t);
RSS = sum((t-hat_t).^2);
R_squared = 1 - RSS / sum((t-bar_t).^2)

% 3. Regularization

% If we need to mitigate over-fitting effects in a model we might resort to
% some regularization techniques, like Ridge regression or Lasso
% regression. 

lambda = 10^(-10);
ridge_coeff = ridge(t, Phi, lambda)

[lasso_coeff, lasso_fit] = lasso(Phi, t)

% where the ridge regression returns only the coefficients in this case w0
% and w1, and the lasso one provides you solutions for different values of
% the regularization paramter lambda.
