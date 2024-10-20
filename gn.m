function [beta,S,betavec]=gn(DTD_FT,H_FT,HTH_FT,b_folder1,b_folder2,b_folder3,xF_folder1,...
    xF_folder2,xF_folder3,beta_0,maxit,tol,c,tau,alpha_0,type,sigma1,sigma2,sigma3)

if ~ismember(type, [1, 2, 3])
    disp('This type is not avaiable')
    return
end

beta=beta_0;


xF1 = dir(fullfile(xF_folder1,'*.jpg'));
xF2 = dir(fullfile(xF_folder2,'*.jpg'));
xF3 = dir(fullfile(xF_folder3,'*.jpg'));

b1 = dir(fullfile(b_folder1,'*.jpg'));
b2 = dir(fullfile(b_folder2,'*.jpg'));
b3= dir(fullfile(b_folder3,'*.jpg'));

for i=1:maxit
    alpha=alpha_0;
    [Jac,f,Sgrad]=Jacfun(DTD_FT,H_FT,HTH_FT,beta,b_folder1,b_folder2,...
        b_folder3,xF_folder1,xF_folder2,xF_folder3,type,sigma1,sigma2,sigma3);
    d=-(Jac'*f)/(Jac'*Jac);
    m=Sgrad'*d;
    if norm(d)<tol
        break
    end
    t=-c*m;
    beta_new=beta+alpha*d;
    f1=norm(f)^2;
    [~,f,~]=Jacfun(DTD_FT,H_FT,HTH_FT,beta_new,b_folder1,b_folder2,b_folder3...
        ,xF_folder1,xF_folder2,xF_folder3,type,sigma1,sigma2,sigma3);
    f2=norm(f)^2;
    while f1-f2<alpha*t
        alpha=tau*alpha;
        beta_new=beta+alpha*d;
        [~,f,~]=Jacfun(DTD_FT,H_FT,HTH_FT,beta_new,b_folder1,b_folder2,b_folder3...
            ,xF_folder1,xF_folder2,xF_folder3,type,sigma1,sigma2,sigma3);
        f2=norm(f)^2;
    end
    
    beta=beta_new;
    betavec(i)=beta;
    S(i)=f2;
end


end
