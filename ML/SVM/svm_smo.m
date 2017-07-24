function [model] = svm_smo(X, Y, C, tol, max_passes)

n_sample = size(X,1);

% Variables
alphas = zeros(n_sample, 1);
b = 0;
epoch = 0;

% Compute the Gram Matrix
K = X * X';

% Train
while epoch < max_passes,
    
    n_updates = 0;
    for i = 1:n_sample
        
        % Calculate the prediction error of the SVM (Ei = f(x(i)) - y(i)).
        E_i = b + sum(alphas .* Y .* K(:,i)) - Y(i);
        
        % Check if the i-th sample is violating a KKT condition
        if ((Y(i) * E_i < -tol && alphas(i) < C) || (Y(i) * E_i > tol && alphas(i) > 0)),
            
            %Select j randomly
            j = randsample([1:i-1 i+1:n_sample],1);
            
            % Calculate Ej = f(x(j)) - y(j) using (2).
            E_j = b + sum(alphas .* Y .* K(:,j)) - Y(j);
            
            % Save old alphas
            alpha_i_old = alphas(i);
            alpha_j_old = alphas(j);
            
            % Compute L and H
            if (Y(i) == Y(j)) %same output
                L = max(0, alphas(j) + alphas(i) - C);
                H = min(C, alphas(j) + alphas(i));
            else % different output
                L = max(0, alphas(j) - alphas(i));
                H = min(C, C + alphas(j) - alphas(i));
            end
            
            if (L ~= H) %if there is something to minimize
                
                % Compute eta (derivative along the constraint)
                eta = K(i,i) + K(j,j) - 2 * K(i,j);
                if (eta > 0) % eta is zero if x_i == x_j
                    
                    % Compute and clip new value for alpha j using (12) and (15).
                    alphas(j) = alphas(j) + (Y(j) * (E_i - E_j)) / eta;
                    alphas(j) = min (H, alphas(j));
                    alphas(j) = max (L, alphas(j));
                    
                    % Check if change in alpha_j is significant
                    if (abs(alphas(j) - alpha_j_old) > tol),
                        
                        % Determine value for alpha i
                        alphas(i) = alphas(i) + Y(i)*Y(j)*(alpha_j_old - alphas(j));
                        % Compute b1 and b2
                        b1 = b - E_i ...
                            - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                            - Y(j) * (alphas(j) - alpha_j_old) *  K(i,j)';
                        b2 = b - E_j ...
                            - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                            - Y(j) * (alphas(j) - alpha_j_old) *  K(j,j)';
                        
                        % Compute b
                        if (0 < alphas(i) && alphas(i) < C),
                            b = b1;
                        elseif (0 < alphas(j) && alphas(j) < C),
                            b = b2;
                        else
                            b = (b1 + b2) / 2;
                        end
                     
                        n_updates = n_updates + 1;
                    
                    end
                end                
            end
        end
    end
    
    if (n_updates == 0),
        epoch = epoch + 1;
    else
        epoch = 0;
    end
end

% Model creation
idx = alphas > 0;
model.x = X(idx,:);
model.y = Y(idx);
model.b = b;
model.alphas = alphas(idx);
model.w = ((alphas .* Y)' * X)';
model.objective = sum(alphas) - 1 / 2 * (model.alphas .* model.y)' * (model.x * model.x') * (model.alphas .* model.y);

end