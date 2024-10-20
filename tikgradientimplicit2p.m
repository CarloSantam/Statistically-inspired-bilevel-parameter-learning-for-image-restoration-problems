function [grad,fun,ss1,ss2,resf,resf2]=tikgradientimplicit2p(type,H_FT,beta,...
    xstar,xF,bb,maxit,tol,ss01,ss02,sigma1,sigma2,M,sigma_mat,k,l)

mu=exp(beta);
K=@(x) real(ifft2(H_FT.*fft2(x)));
KT=@(x) real(ifft2(conj(H_FT).*fft2(x)));
    
% Differenze rispetto a prima:
% 1) l'hessiana stavolta è H=H^T D_\mu H + D_h^T*D_h + D_v^T * D_v.
% 2) questa volta l'hessiana è definita positiva in senso stretto. Questo
% vuol dire che possiamo usare tranquillamente il conjugate gradient :) 
% Per il resto tutto è identico a prima :)

    n=length(H_FT);
    
    c=K(xstar);
    z=c-bb;
    K1=M.*z;
    %a=mu(:);
    
    k1=k(1);
    
    b1=mu(k1)*KT(K1);
    
    b1=-b1(:);

    K2=(ones(n)-M).*z;
    l1=l(1);
    b2=mu(l1)*KT(K2);
    
    b2=-b2(:);

    [ss1,~,resf]=pcg(@(x)hessian(x,K,KT,n,mu),b1,tol,maxit,[],[],ss01);
    s1=reshape(ss1,[n,n]);
    [ss2,~,resf2]=pcg(@(x)hessian(x,K,KT,n,mu),b2,tol,maxit,[],[],ss02);
    s2=reshape(ss2,[n,n]);
    funn=xstar(:)-xF(:);


if type==1
    grad=[s1(:),s2(:)];
    fun=funn;

elseif type==2
    
    Ress=real(ifft2(H_FT.*fft2(xstar)))-bb;

    RESS=Ress(:);    
       
    a=abs(RESS(k)).^2-sigma1^2;
    
    c=abs(RESS(l)).^2-sigma2^2;
    
    r1=sum(a);
    
    r2=sum(c);
    
    gder1=ifft2(H_FT.*fft2(s1));
    
    gradd1=2.*Ress.*gder1;

    gradd1=gradd1(:);
        
    grad11=sum(gradd1(k))
    
    grad21=sum(gradd1(l))
    
    %grad21=0;

    gder2=ifft2(H_FT.*fft2(s2));
    
    gradd2=2.*Ress.*gder2;

    gradd2=gradd2(:);

    grad12=sum(gradd2(k));
    
    %grad12=0;
    
    grad22=sum(gradd2(l));
    
    grad=[grad11,grad12;grad21,grad22];
    
    fun=[r1;r2];
    
    %{
    grad=real(ifft2(H_FT.*fft2(s)));
    Res=real(ifft2(H_FT.*fft2(xstar)))-bb;
    grad=grad(:);
    Res=Res(:);
    grad=2*grad'*Res;
    fun=norm(Res)^2-(sigma^2)*n^2;
    %}
elseif type==3
    
    gder1=sigma_mat.*real(ifft2(H_FT.*fft2(s1)));
    Ress=sigma_mat.*real((ifft2(H_FT.*fft2(xstar))))-sigma_mat.*bb;
    
    ccorrelation1_1=real(ifft2(fft2(gder1).*conj(fft2(Ress))))+real(ifft2(fft2(Ress).*conj(fft2(gder1))));
    ccorrelation=real(ifft2(fft2(Ress).*conj(fft2(Ress))));
    gder1=gder1(:);
    Resnorm=norm(Ress,'fro');
    resgrad=2*gder1'*Ress(:);
    fun=ccorrelation(:)/Resnorm^2;
    grad1=(ccorrelation1_1(:)*Resnorm^2-ccorrelation(:)*resgrad)/Resnorm^4;
    
    gder2=sigma_mat.*real(ifft2(H_FT.*fft2(s2)));
    ccorrelation1_2=real(ifft2(fft2(gder2).*conj(fft2(Ress))))+real(ifft2(fft2(Ress).*conj(fft2(gder2))));
    gder2=gder2(:);
    resgrad=2*gder2'*Ress(:);
    grad2=(ccorrelation1_2(:)*Resnorm^2-ccorrelation(:)*resgrad)/Resnorm^4;
    
    grad=[grad1(:),grad2(:)];
end
    
end


function Dhu = Dh(u)
    Dhu  = [ u(:,2:end) - u(:,1:(end-1)) , u(:,1) - u(:,end) ];
end
function Dvu = Dv(u)
    Dvu  = [ u(2:end,:) - u(1:(end-1),:) ; u(1,:) - u(end,:) ]; 
end
function DhTu = DhT(u)
    DhTu = [ u(:,end) - u(:,1) , -diff(u,1,2) ];
end
function DvTu = DvT(u)
    DvTu = [ u(end,:) - u(1,:) ; -diff(u,1,1) ];
end

function HE = hessian(x,K,KT,size,mu)

x=reshape(x,[size,size]);
%n=size;
%mu_mat = ones(size);
%mu_mat(1:n,(n/2+1):n)=mu(end);
%mu_mat(1:n,1:n/2)=mu(1);

HE1=mu.*K(x);

HE1=KT(HE1);

%c=c(:);
b=DhT(Dh(x))+DvT(Dv(x));
%b=b(:);
HE=HE1+b;
HE=HE(:);
end
