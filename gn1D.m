function [beta,fk,time,x]=gn1D(DTD_FT,H_FT,HTH_FT,bbhat,xF,beta_0,maxit,tol,type,sigma,n,bb)

beta=beta_0;

time=0;

fk=nan(maxit,1);

% per type=1-->MSE
% per type=2-->Gaussianity
% per type=3-->Whiteness

if type==1

for i=1:maxit
    tstart=tic;
    [grad,f,~,~,x]=gradfun(DTD_FT,H_FT,HTH_FT,beta,bbhat,xF);
    d=-(grad'*f)/(grad'*grad); %calcoliamo il sistema lineare (Jac'*Jac)\(-Jac'*f).
    %NB: qui il parametro è uno solo, il tutto si riduce quindi a una
    %frazione.
    fk(i)=norm(f)^2; % calcoliamo il valore della loss per questo parametro.
    if norm(d)<tol
        break
    end
    beta=beta+d; %svolgiamo una iterazione di Gauss-Newton
    a=toc(tstart);
    time=time+a;
end

elseif type==2

for i=1:maxit
    tstart=tic;
    [grad,f,x]=gradfungauss(DTD_FT,H_FT,HTH_FT,beta,bbhat,n,sigma);
    d=-(grad'*f)/(grad'*grad);
    fk(i)=norm(f)^2;
    if norm(d)<tol
        break
    end
    beta=beta+d;
    a=toc(tstart);
    time=a+time;
end

elseif type==3

for i=1:maxit
    tstart=tic;
    [grad,f,x]=gradfunwhiteness(DTD_FT,H_FT,HTH_FT,beta,bbhat,bb);
    d=-(grad'*f)/(grad'*grad);
    fk(i)=norm(f)^2;
    if norm(d)<tol
        break
    end
    beta=beta+d;
    a=toc(tstart);
    time=a+time;
end

else
    disp("This type is not avaiable")
end
fk=rmmissing(fk);
end
