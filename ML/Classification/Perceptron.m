% Perceptron

load iris_dataset.mat;
x = zscore(irisInputs([1 2],:)');
t = irisTargets(1,:)';
gplotmatrix(x, [], t);

% Let us perform a classification with a perceptron classifier.
% 
% - Hypothesis space: y(x) = sgn(w^Tx)
% - Loss measure: Distance of misclassified points L(w) = -SUM(w^TxC)
% - Optimization method: Online Gradient Descent

% net = perceptron;
% net = train(net, x', t');

% To evaluate the performance of the chosen method we need to consider the
% confusion matrix which tells us the number of points which have been
% correctly classified and those which have been misclassified.cle
% Consequently, we can compute the following metrics:
%
% - Accuracy: Acc = (tp+tn/N) Fraction of the samples correclty classified. 
% - Precision: Pre = tp/(tp+fp) Fraction of samples correctly classified
% in the positive class amoung the ones classified in the positive class
% - Recall: Rec = tp/(tp+fn) Fraction of samples correctly classified in
% the positive class among the ones belonging to the positive class
% - F1 score: F1 = (2*Pre * Rec) / (Pre + Rec) Harmonic mean of precision
% and recall

% Let's implement the perceptron by hand:
% We first convert the output to be {-1,1} and we permuted the samples not
% to process all the samples from a single class consecutivrely. Remember
% that while the NN toolbox is able to stop the optimization process based
% on the classification error, here we need to perform a large enough
% number of stochastic gradient updates s.t. it is able to correctly
% classify all the samples.

n_samples = size(x,1);
perc_t = t;
perc_t(perc_t == 0) = -1;
ind = randperm(n_samples);

perc_t = perc_t(ind);
x_perc = x(ind,:);

w = ones(1,3);
for jj = 1:10
    for ii = 1:n_samples
        if sign(w * [1 x_perc(ii,:)]') ~= perc_t(ii)
            w = w + [1 x_perc(ii,:)] * perc_t(ii);
        end
        if mod(ii,50) == 0
            plot(s, -(s * w(2) + w(1) ) / w(3), 'r');
        end
    end
end

% Finally, let's plot the learned separation surface
figure();
gscatter(x_perc(:,1),x_perc(:,2),perc_t);
hold on;
axis manual;
s = -2:0.01:2.5;
h(1) = plot(s, -(s * net.IW{1}(1) + net.b{1} ) / net.IW{1}(2),'k');
h(2) = plot(s, -(s * w(2) + w(1) ) / w(3),'r');
legend(h,{'Perceptron (func)' 'Perceptron (hand)'})