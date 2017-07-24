function K = gaussKer(xi, xj, theta)

n = size(xi, 1);
m = size(xj, 1);
K = zeros(n,m);
for ii = 1:n
    for jj = 1:m
       K(ii,jj) = exp(2 * theta(2)) * exp( - (xi(ii,:) - xj(jj,;)).^2 / (2 * exp(2 * theta(1))));
    end
end
