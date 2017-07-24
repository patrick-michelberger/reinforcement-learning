% Bias-Variance Tradeoff

% Do deal with the Bias-Variance dilemma we use a set of techniques able to
% take into account the bias / variance tradeoff without having the
% necessity to know the real model.

% First we generate a new dataset

n_tot = 40;
eps = 2;
func = @(x)((0.5 - x) .* (5 - x) .* (x -3));

x = 5 * rand(n_tot,1);
y = func(x) + eps * randn(n_tot, 1);

% We divide the dataset into:
% 
% - Training set: the data we will use to learn the model parameters 
% - Validation set: the data we will use to select the model
% - Training set: the data we will use to test the performance of our model

n_train = 20;
n_valid = 10;
n_test = 10;

x_train = x(1:n_train);
y_train = y(1:n_train);

x_valid = x(1:n_valid);
y_valid = y(1:n_valid);

x_test = x(1:n_test);
y_test = y(1:n_test);

figure();
plot(x_train, y_train, '.');
hold on;
plot(x_test, y_test, '.');
plot(x_valid, y_valid, '.');
plot(linspace(0,5,100), func(linspace(0,5,100)));
ylabel('Target');
xlabel('Input');
legend({'Training set' 'Validation set' 'Test set' 'Real function'});

% Here we are not required to shuffle the data since we generate the input
% at random, but in general it is a good practice to randomly rearrange the
% data before splitting, since some ordering might be induced by those who
% provided you the data.

% Hypothesis space: y = f(x,w) = SUM_k(x^k*w)
% Loss measure: RSS
% Optimization method: LS method

% We want to select among polynomial models with order {0,...9}.

for order = 0:9
    lin_model{order+1} = fitlm(x_train, y_train, ['poly' num2str(order)]);
    MSE(order+1) = lin_model{order+1}.MSE;
end

figure();
plot(MSE);
title('Only training');
xlabel('Model parameters');
ylabel('MSE');

% We would choose the model with order 9 or on average the most complex one
% or the one with lowest bias and highest variance. We can see that this
% model does not equally perform on a test set.

for order = 0:9
    y_pred = predict(lin_model{order+1}, x_test);
    MSE_test(order+1) = mean((y_pred - y_test).^2);
end

figure();
plot(MSE);
hold on;
plot(MSE_test);
legend({'Training MSE' 'Test MSE'});
title('Training and test error');
xlabel('Model parameters');
ylabel('MSE');

% To deal with this problem we will consider a validation set and tune the 
% order o of the polynomial on that set:

for order = 0:9
     y_pred = predict(lin_model{order+1}, x_valid);
    MSE_valid(order+1) = mean((y_pred - y_valid).^2);
end

figure()
plot(MSE);
hold on;
plot(MSE_test);
plot(MSE_valid);
[y_min, x_min] = min(MSE_valid);
plot(x_min,y_min,'x');
legend({'Training MSE' 'Test MSE' 'Validation MSE'});
xlabel('Model parameters');
ylabel('MSE');

% However, the results of this procedure are strongly dependent on the 
% validation set we used. Moreover, we are not using some of the samples 
% during the training, which could improve the model accuracy. 
% To better exploit the available data, we could resort to the use of 
% crossvalidation. This way we have to join the Training and Validation set 
% and divide it in k equally long partitions and use sequentially k - 1 of 
% them as training set and the remaining one as validation set. 
% In the end we average the results obtained this way. 

% Considering k = 4 we have:

k_fold = 4;
x_cross = [x_train; x_valid];
y_cross = [y_train; y_valid];
n_cross = n_train + n_valid;

for order = 0:9
    MSE_cross(order+1) = 0;
    % Divide data
    for kk = 1:k_fold
        ind_valid = 1 + round(n_cross * (kk - 1) / k_fold ) : round(n_cross * kk / k_fold );

        x_valid = x_cross(ind_valid);
        y_valid = y_cross(ind_valid);
        x_train = x_cross; x_train(ind_valid) = [];
        y_train = y_cross; y_train(ind_valid) = [];

    % Fit model
    model = fitlm(x_train, y_train, ['poly' num2str(order)]);
    y_pred = predict(model, x_valid);
    MSE_cross(order+1) = MSE_cross(order+1) + mean((y_pred - y_valid).^2);
    end
    
    MSE_cross(order+1) = MSE_cross(order+1) / k_fold;
end
toc

figure()
h(1) = plot(MSE);
hold on;
h(2) = plot(MSE_test);
h(3) = plot(MSE_valid);
[y_min, x_min] = min(MSE_valid);
plot(x_min,y_min,'x');
h(4) = plot(MSE_cross);
[y_min, x_min] = min(MSE_cross);
plot(x_min,y_min,'x');
legend(h,{'Training MSE' 'Test MSE' 'Validation MSE' 'Crossvalidation MSE'},'location','northwest');
xlabel('Model parameters');
ylabel('MSE');

% You can see that the value of the error in the optimal point is greater 
% than the actual error we have on a test set.

% If we want to perform the Leave One Out (LOO), also called Jackknife, 
% we just have to modify the number of folds we consider in the 
% cross-validation procedure, i.e., k_fold = 40;. This kind of procedure is 
% less biased than cross-validation, but requires more computational time.