function h = plotGp(GPmodel)

x_train = GPmodel.X;
t_train = GPmodel.Y;

x_new = linspace(min(x_train), max(x_train), 1000);
[t_new, ~, t_int] = predict(GPmodel, x_new');


h = figure();
hold on;
conf_area = fill([x_new x_new(end:-1:1)],[t_int(:,1)' t_int(end:-1:1,2)'], 'k');
alpha(conf_area, 0.2);
plot(x_new,t_new, 'b', 'LineWidth', 2);
plot(x_train, t_train, 'r.');
axis tight;
title(['$\theta =$ [' num2str(GPmodel.KernelInformation.KernelParameters') '], $\sigma =$ ' num2str(GPmodel.Sigma) ],'interpreter', 'Latex');
