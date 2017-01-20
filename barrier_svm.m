function [fs, w] = barrier_svm(X,y,alpha,beta)
    C = 1000;

    y(y==0) = -1;

    % Calculate RBF
    sig = 1000;
    n = size(X, 1);
    K = X*X'/sig^2;
    d = diag(K);
    K = K-ones(n,1)*d'/2;
    K = K-d*ones(1,n)/2;
    K = exp(K);
    
    % n * n
    K_til = y*y'.*K;
    
    mu = 1.5;

    % Find start feasible point
    f = [zeros(n,1);1];
    AIneq = [eye(n),-ones(n,1);...
             -eye(n),-ones(n,1)];
    bIneq = [C*ones(n,1);zeros(n,1)];
    x = linprog(f,AIneq,bIneq);
    w = x(1:n);    
    
    w = 500*ones(n,1);

    grad = @(t,w) (K_til*w - 1) + 1/t*(1./(C-w) - 1./w);
    Hessian = @(t,w) K_til + 1/t*(diag(1./(C-w).^2 + 1./w.^2));
    A = @(t,w) [Hessian(t,w), y;y',0];
    B = @(t,w) [grad(t,w);y'*w];
    
    f = @(w) 0.5*w'*K_til*w-sum(w);
    g = @(t,w) f(w) - (1/t)*sum(log(w.*(C-w)));
    newton_decrement = @(t,w,deltaW) deltaW'*Hessian(t,w)*deltaW;
    
    k = 0;
    t_outer = 1000;
    fs = [f(w)];
    
    while 2*n/t_outer > 1e-8
        disp(t_outer);
        gs = [g(t_outer,w)];
        while true
            v = -A(t_outer,w)\B(t_outer,w);
            deltaW = v(1:n);
            
            tmp = -w./deltaW;
            t_max = min(1, min(tmp(deltaW<0)));
            t_inner = t_max*0.99;
            
            t_inner = 1;

            tmp = alpha*grad(t_outer,w)'*deltaW;
            
            while g(t_outer,w+t_inner*deltaW)>g(t_outer,w)+ t_inner*tmp ...
                   || sum(w+t_inner*deltaW >= C) ~= 0 || ...
                   sum(w+t_inner*deltaW <= 0) ~= 0
                t_inner = beta*t_inner;
            end
                
            w = w + t_inner*deltaW;

            fs = [fs f(w)];
            
            if newton_decrement(t_outer,w,deltaW) <= 10e-6
                break
            end
            
            k = k + 1;
        end
        t_outer = mu*t_outer;
    end
%     f_diff = fs - f_optimal;
end
