function [fs,w] = primal_dual_svm(X,y,alpha,beta)
    C = 1000;
    mu = 50;
    
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
    
    t = @(w,u,v) -2*n*mu/([-w;w-C]'*[u;v]);
    
    % Find start feasible point
    f = [zeros(3*n,1);1];
    AIneq = [eye(n),zeros(n,2*n),-ones(n,1);...
             -eye(n),zeros(n,2*n),-ones(n,1);...
             zeros(n,n),-eye(n),zeros(n,n),-ones(n,1);...
             zeros(n,2*n),-eye(n),-ones(n,1)];
    bIneq = [C*ones(n,1);zeros(3*n,1)];
    
    x = linprog(f,AIneq,bIneq);
    w = x(1:n);
    u = x(n+1:2*n);
    v = x(2*n+1:3*n);
    u = ones(n,1);
    v = ones(n,1);
    lambda = 0;

    A = @(w,u,v,lambda) [K_til,-eye(n),eye(n),y;...
                        -diag(u),diag(-w),zeros(n,n+1);...
                        diag(v),zeros(n,n),diag(w-C),zeros(n,1);y',zeros(1,2*n+1)];
    
    r_prim = @(w,u,v,lambda) K_til*w-1-u+v+lambda*y;
    r_cent = @(w,u,v) [diag(u)*(-w);diag(v)*(w-C)]+1/t(w,u,v);
    r_dual = @(w) y'*w;

    B = @(w,u,v,lambda) [r_prim(w,u,v,lambda);r_cent(w,u,v);r_dual(w)];
    
    f = @(w) 0.5*w'*K_til*w-sum(w);
    
    deltaZ = @(w,u,v,lambda) linsolve(A(w,u,v,lambda),-B(w,u,v,lambda));
    update = deltaZ(w,u,v,lambda);
    
    tmp = update(n+1:3*n);
    tmp2 = -[u;v]./tmp;
    theta_max = min(1, min(tmp2(tmp<0)));

    theta = 0.99*theta_max;

    fs = [f(w)];
    while true
        disp(1)
        update = deltaZ(w,u,v,lambda);
        
        w_new = w+theta*update(1:n);
        u_new = u+theta*update(n+1:2*n);
        v_new = v+theta*update(2*n+1:3*n);
        lambda_new = lambda+theta*update(end);
        
        while [-w_new;w_new-C] >= 0
            theta = beta*theta;
            disp(theta);
        end
                
        r_new = B(w_new,u_new,v_new,lambda_new);
        r = B(w,u,v,lambda);
        
        while norm(r_new) > (1-alpha*theta)*norm(r)
            theta = beta*theta;
            disp(1)
        end
        
        rp = r_prim(w,u,v,lambda);
        rd = r_dual(w);
        
        if sqrt(sum(rp.^2)+sum(rd.^2)) <= 10e-6 || -[-w;w-C]'*[u;v] <= 10e-6
            disp(-[-w;w-C]'*[u;v]);
            break
        end
        
        w = w_new;
        u = u_new;
        v = v_new;
        lambda = lambda_new;
        disp(theta);
        fs = [fs f(w)];
    end
end