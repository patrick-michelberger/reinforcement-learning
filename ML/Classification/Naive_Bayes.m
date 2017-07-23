% Naive Bayes

clear 
clc
close all

load iris_dataset.mat;
x = zscore(irisInputs([1 2],:)');
t = irisTargets(1,:)';

% Let us use a generative model: Naive Bayes. Generative models have the
% purpose of modeling the joint pdf of the couple input / output p(C,x)
% which allows us to generate also new data from what we learned,
% differently from discriminative models we are only interested in
% computing the probabilities that a given input is coming from a specific
% class p(C?|?x), which is not suffiecient to product new samples.

% In this case the NB method considers the naive assumption that each input
% is conditionally independent! (w.r.t. the class) from each other. If we
% consider the Bayes formula we have:
%
% p(C|x) = (p(C) * p (x|C)) / p(x) = p(C) * PROD(p(X|C))

% The decision functino, which maximizes the MAP probability is 
% 
% y(x) = arg max p(C|x)
% 
% In a specific case we have to define a prior distribution for the classes
% P(C) and a distribution to compute the likelihood of the considered
% samples P(x|C). In the case of continuous variable one of the usual
% assunmption is to use Gaussian distribution for each variable p (x|C) and
% either a uniform prior p(C)  = 1 / K or a multinomial prios based on the
% samples proportions P(C).
%
% The complete model we consider is:
%
% - Hypothesis space: arg max p(C|x)
% - Loss function: Log likelihood
% - Optimization method: MLE

% By considering the trained parameters we are able to generate new data.
% For each data we randomly pick a class, according to the class prior and
% then sample from the corresponding Gaussian distribution for each
% dimension of the data n_dim

nb_model = fitcnb(x,t);

t_pred = predict(nb_model,x);
confusionmat(t,t_pred)

figure();
gscatter(x(:,1),x(:,2),t);
hold on;
axis manual

[a, b] = meshgrid(-3:0.1:3,-3:0.1:4);
axis tight;
pred = predict(nb_model,[a(:),b(:)]);
gscatter(a(:),b(:),pred);
  
%% Generative abilities of NB
param = nb_model.DistributionParameters;
prior = cumsum(nb_model.Prior);
n_dim = size(param,2);

n_gen = 1000;
gendata = zeros(n_gen,n_dim);
gentarget = zeros(n_gen,1);

for ii = 1:n_gen
    gentarget(ii) = find(prior > rand(),1);
    for jj = 1:n_dim
        mu = param{gentarget(ii),jj}(1);
        sigma = param{gentarget(ii),jj}(2);
        gendata(ii,jj) = normrnd(mu,sigma);
    end
end

figure();
gscatter(gendata(:,1),gendata(:,2),gentarget);
