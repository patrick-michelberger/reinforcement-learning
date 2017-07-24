% Support Vector Machines 

% In this exercise we will explore the possible options we have when we
% consider as classification tool the Support Vector Machines.

clear
clc
close all

% First we load the data corresponding to the two considered classes and
% set the targets.

load iris_dataset;

irisInputs = zscore(irisInputs(1:2,1:100)');
irisTargets = [ones(50,1); -ones(50,1)];

figure();
gplotmatrix(irisInputs,[],irisTargets);

% As usual we normalize the data. Since we are going to use SVM we have to
% perform this operation. Otherwise it would change the final solution and
% influence the convergence rate. 

% We resort to a linear SVM 
%
% - Hypothesis space: y = f(x,w) = sign(W^Tx + b)
% - Loss measure: |w|^2 + C * SUM(c) where (w^Tx + b) >= 1 - c for all n
% - Optimization method: Quadratic optimization

% We use the fitcsvm method which considers a linear SVM without kernel and
% C=1

svm_model = fitcsvm(irisInputs, irisTargets);

% With the option verbose we are able to check the iterations performed
% during the optimization procedure, for instance by inspecting the number
% of support vectors during the procedure or the total number of violation
% of the constraints SUM(c). If we want to visualize the classfication
% boundary induced by the SVM, we should extract the model parameters w
% and the bias b stored in the model

w = svm_model.Beta;
b = svm_model.Bias;

% We finally visualize the boundary between the two classes and the margins

figure();
pos_class = irisTargets == 1;
neg_class = irisTargets == -1;
plot(irisInputs(pos_class,1), irisInputs(pos_class,2),'r.');
hold on;
plot(irisInputs(neg_class,1), irisInputs(neg_class,2),'g.');

% Classes bound
x = min(irisInputs(:,1)):0.1:max(irisInputs(:,1));
y = -w(1) / w(2) * x - b / w(2);
plot(x,y);

%Margins
y = -w(1) / w(2) * x + (1 - b) / w(2);
plot(x,y,'--');
y = -w(1) / w(2) * x + (-1 - b) / w(2);
plot(x,y,'--');
xlabel('x_1');
ylabel('x_2');
axis tight

M = 1 / norm(w)

% We can see that the boundaries are linear and thanks to soft margins
% (given by the use of the slack variables) we have some instances of the
% training set between the margins.

% It is also possible to extract the support vectors from the dataset we
% considered and plot them

support_vec = svm_model.SupportVectors;
plot(support_vec(:,1), support_vec(:,2),'rx');

% The support vectors are the only one contributing directly to the SVM
% parameters w. They are clearly placed inside the box or on the margins.

% If we artificially manipulate the input space s.t. we have different
% scales on the two axes, the SVM behaves in a different way. 

% Let's now try to add a Kernel to the SVM, for instance a Gaussian kernel.
% In this case, an equivalent version of the perceptron classifier would
% require the input feature space to infinite dimensions. We resort to the
% kernel trick and we only need to compute the Gram matrix K (N x N) 

svm_kernel_model = fitcsvm(irisInputs, irisTargets, 'KernelFunction', 'gaussian');

% To visualize the new separating hyperplane, which is linear in the kernel
% space but have arbitrary shape in the input space, we need to plot the
% contour of the function w^T * k(x) + b

C = 1;
bandwidth = 0.5;

% Visualize the Kernel SVM
min_x = min(irisInputs);
max_x = max(irisInputs);
n_p = 50;
x = linspace(min_x(1),max_x(1),n_p);
y = linspace(min_x(2),max_x(2),n_p);
[X,Y] = meshgrid(x,y);
[~,z] = predict(svm_kernel_model,[X(:) Y(:)]);
Z = reshape(z(:,1),n_p,n_p) / 50;

figure();
pos_class = irisTargets == 1;
neg_class = irisTargets == -1;
plot(irisInputs(pos_class,1), irisInputs(pos_class,2),'r.');
hold on;
plot(irisInputs(neg_class,1), irisInputs(neg_class,2),'g.');
contour(X,Y,Z);
xlabel('x_1');
ylabel('x_2');
title(['C = ' num2str(C) ', l = ' num2str(bandwidth)]);
axis tight

support_vec = svm_kernel_model.SupportVectors;
plot(support_vec(:,1), support_vec(:,2),'rx');

% Solve the dual form with SMO

manual_svm = svm_smo(irisInputs,irisTargets,1,0.0001,4)

w = manual_svm.w;
figure();
plot(irisInputs(pos_class,1), irisInputs(pos_class,2),'r.');
hold on;
plot(irisInputs(neg_class,1), irisInputs(neg_class,2),'g.');

% Classes bound
x = min(irisInputs(:,1)):0.1:max(irisInputs(:,1));
y = -w(1) / w(2) * x - b / w(2);
plot(x,y);

%Margins
y = -w(1) / w(2) * x + (1 - b) / w(2);
plot(x,y,'--');
y = -w(1) / w(2) * x + (-1 - b) / w(2);
plot(x,y,'--');
xlabel('x_1');
ylabel('x_2');
axis tight