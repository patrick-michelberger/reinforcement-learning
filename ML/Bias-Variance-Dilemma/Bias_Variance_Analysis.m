% Bias-Variance Analysis

% First, we generate a synthetic dataset and examine the phenomenon of bias
% / variance tradeoff in different models. After that we will analyse some
% technique in order to manage the trade-off on real data. The presented
% procedures allow to decide which model, among a set of given models, is
% best suited for the problem analysed. For instance, we might want to
% select the most important features to use, the order of the polynomial in
% a regression, the coefficient for the regularization term or the K
% parameter in the K-NN method.

n_points = 1000;
eps = 0.7;
func = @(x) (1 + 1/2 * x + 1 / 10 * x.^2);

x = 5 * rand(n_points,1);
t = func(x);
t_noisy = func(x) + eps * rand(n_points, 1);

% After that we consider two different linear regression models.

lin_model = fitlm(x, t_noisy);

phi = [x x.^2];
qua_model = fitlm(phi, t_noisy);

% Let us plot in the parameter space the models we estimated and the
% optimal ones.

real_par = [1 1/2 1/10];
best_lin_par = [7/12 1 0];

lin_c = [lin_model.Coefficients.Estimate' 0];
qua_c = qua_model.Coefficients.Estimate;

figure();
plot3(real_par(1),real_par(2),real_par(3),'bx');
hold on
grid on;
plot3(best_lin_par(1),best_lin_par(2),best_lin_par(3),'ro');
plot3(lin_c(1),lin_c(2),lin_c(3),'r+');
plot3(qua_c(1),qua_c(2),qua_c(3),'b+');
title('Parameter space');
xlabel('w_0');
ylabel('w_1');
zlabel('w_2');

% As you can see from the two realizations of the parameters, the
% parameters of the family L1 lies in a 2D subspace, while the approximated
% quadratic models may span over a 3D subspace. Hence models from L2 might
% coincide with the real model, while linear models coming from L1 will
% always suffer from a bias which can not be reduced to zero.

% If we want to evaluate the bias and variance of the two chosen models, we
% should iterate the estimation procedure over multiple datasets.

n_repetitions = 100;
for ii = 1:n_repetitions
    % sample generation
    x = 5 * rand(n_points, 1);
    t = func(x);
    t_noisy = func(x) + eps * randn(n_points,1);
    phi = [x x.^2];
    
    lin_model = fitlm(x, t_noisy);
    qua_model = fitlm(phi, t_noisy);
    
    lin_coeff(ii,:) = [lin_model.Coefficients.Estimate' 0];
    qua_coeff(ii,:) = qua_model.Coefficients.Estimate;
end

figure();
plot3(real_par(1),real_par(2),real_par(3),'bx');
hold on
grid on;
plot3(best_lin_par(1),best_lin_par(2),best_lin_par(3),'ro');
plot3(lin_coeff(:,1),lin_coeff(:,2),lin_coeff(:,3),'r.');
plot3(qua_coeff(:,1),qua_coeff(:,2),qua_coeff(:,3),'b.');

title('Parameter space');
xlabel('w_0');
ylabel('w_1');
zlabel('w_2');

% In the figure you see how different trained models spread around the
% optimal parameters. The more points we consider for estimation, the more
% the resulting parameter vectors are close to the real and best model in
% hypothesis space L1, depending on the hypothesis space we choose.
% Remember that event if the estimated model coincides with the best model,
% we are not able to reduce the error on newly seen points to zero due to
% the irreducible error sigma^2.

% At last, we extract a new point and evaluate its bias and variance.

x_new = 5 * rand();
t_new = func(x_new) + eps * randn(1,1);
x_enh_new = [1 x_new 0];
phi_new = [1 x_new x_new.^2];

for ii = 1:n_repetitions
    y_pred_lin(ii) = lin_coeff(ii,:) * x_enh_new';
    y_pred_qua(ii) = qua_coeff(ii,:) * phi_new';
end

error_lin = mean((t_new - y_pred_lin).^2);
bias_lin = mean(func(x_new) - y_pred_lin)^2;
variance_lin = var(y_pred_lin);
var_t_lin = error_lin - variance_lin - bias_lin;

error_qua = mean((t_new - y_pred_qua).^2);
bias_qua = mean(func(x_new) - y_pred_qua)^2;
variance_qua = var(y_pred_qua);
var_t_qua = error_qua - variance_qua - bias_qua;

disp('---Single point---');
disp(['Linear error: ' num2str(error_lin)]);
disp(['Linear bias: ' num2str(bias_lin)]);
disp(['Linear variance: ' num2str(variance_lin)]);
disp(['Linear sigma: ' num2str(var_t_lin) ' (unreliable)']);

disp(['Quadratic error: ' num2str(error_qua)]);
disp(['Quadratic bias: ' num2str(bias_qua)]);
disp(['Quadratic variance: ' num2str(variance_qua)]);
disp(['Quadratic sigma: ' num2str(var_t_qua) ' (unreliable)']);

% In this case the estimation of the variance sigma^2 does not make sense
% since it has been done on a single sample.

% Similiarly we evaluate bias and variance by integrating over the entire
% input space [0,5] and have a more stable estimate.

n_samples = 101;
x_new = linspace(0,5,n_samples)';
t_new = func(x_new) + eps * randn(n_samples,1);
x_enh_new = [ones(n_samples,1) x_new zeros(n_samples,1)];
phi_new = [ones(n_samples,1) x_new x_new.^2];

for ii = 1:n_repetitions    
    y_pred_lin_all(ii,:) = lin_coeff(ii,:) * x_enh_new';
    y_pred_qua_all(ii,:) = qua_coeff(ii,:) * phi_new';
end

error_lin = sum(mean((repmat(t_new',n_repetitions,1) - y_pred_lin_all).^2)) / n_samples;
bias_lin = sum(mean(repmat(func(x_new'),n_repetitions,1) - y_pred_lin_all).^2) / n_samples;
variance_lin = sum(var(y_pred_lin_all)) / n_samples;
var_t_lin = (error_lin - bias_lin - variance_lin);

error_qua = sum(mean((repmat(t_new',n_repetitions,1) - y_pred_qua_all).^2)) / n_samples;
bias_qua = sum(mean(repmat(func(x_new'),n_repetitions,1) - y_pred_qua_all).^2) / n_samples;
variance_qua = sum(var(y_pred_qua_all)) / n_samples;
var_t_qua = (error_qua - bias_qua - variance_qua);

disp('---All dataset---')
disp(['Linear error: ' num2str(error_lin)]);
disp(['Linear bias: ' num2str(bias_lin)]);
disp(['Linear variance: ' num2str(variance_lin)]);
disp(['Linear sigma: ' num2str(var_t_lin)]);

disp(['Quadratic error: ' num2str(error_qua)]);
disp(['Quadratic bias: ' num2str(bias_qua)]);
disp(['Quadratic variance: ' num2str(variance_qua)]);
disp(['Quadratic sigma: ' num2str(var_t_qua)]);

% With this procedure we are able to check that the L2 is generally able to
% reduce both bias and variance, but it is not able to get a null irreducible error.