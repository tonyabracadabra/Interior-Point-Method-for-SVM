function [K] = RBF(X1,X2)
    sigma = 1000;
    X1 = X1(:); X2 = X2(:);
    xny = X1-X2;
    Normxny = xny'*xny;
    K = exp(-Normxny/(2*sigma^2));
end