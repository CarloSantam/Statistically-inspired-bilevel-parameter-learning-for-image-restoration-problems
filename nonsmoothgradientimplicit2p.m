function [grad,fun,ss1,ss2,resf,resf2]=nonsmoothgradientimplicit2p(type,epsi,H_FT,beta,...
    xstar,xF,bb,maxit,tol,ss01,ss02,sigma1,sigma2,M,sigma_mat,k,l)

mu=exp(beta);
K=@(x) real(ifft2(H_FT.*fft2(x)));
KT=@(x) real(ifft2(conj(H_FT).*fft2(x)));
    
    Dhu=Dh(xstar);
    Dvu=Dv(xstar);
    
    dd1=huber_2(Dhu,epsi);
    dd2=huber_2(Dvu,epsi);
    
    n=length(H_FT);
    
    c=K(xstar);
    z=c-bb;
    K1=M.*z;
    a=mu(:);
        
    b1=a(k(1))*KT(K1); 
    
    b1=-b1(:); %calcoliamo l'opposto additivo della derivata del gradiente 
    % rispetto a beta_1 * la derivata rispetto a \beta_1 di e^\beta
    % Ricordiamo che
    % - \beta, questa volta è un vettore 2x1, anche se per
    % comodità viene considerato come una immagine n x n, uguale a \beta_1 in 
    % \Omega_1, e \beta_2 in \Omega_2).
    % - M è la maschera di \Omega_1, quindi 1 in \Omega_1, 0 fuori.
    % - k=find(M==1) e l=find(M==0). Perciò a(k(1))=\mu_1=exp(\beta_1) e
    % a(l(1))=\mu_2=exp(\beta_2).

    K2=(ones(n)-M).*z;
    b2=a(l(1))*KT(K2); 
    b2=-b2(:);

    [ss1,~,resf]=minres(@(x)hessian(x,K,KT,n,dd1,dd2,mu),b1,tol,maxit,[],[],ss01);
    s1=reshape(ss1,[n,n]);
    %esattamente come prima andiamo ad utilizzare il minres e non cg, visto
    %che l'hessiana NON è definita positiva, ma solo semidefinita positiva.
    %i due [],[] sono dovuti al fatto che non precondizioniamo.
    [ss2,~,resf2]=minres(@(x)hessian(x,K,KT,n,dd1,dd2,mu),b2,tol,maxit,[],[],ss02);
    s2=reshape(ss2,[n,n]);
    funn=xstar(:)-xF(:);

% questa parte è scritta bene sulla tesi per quanto riguarda i calcoli.
% Ovviamente per necessità di chiarimenti resto disponibile come al solito
% :)
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
        
    grad11=sum(gradd1(k));
    
    grad21=sum(gradd1(l));
    
    gder2=ifft2(H_FT.*fft2(s2));
    
    gradd2=2.*Ress.*gder2;

    gradd2=gradd2(:);

    grad12=sum(gradd2(k));
        
    grad22=sum(gradd2(l));
    
    grad=[grad11,grad12;grad21,grad22];
    
    fun=[r1;r2];
    
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

function HE = hessian(x,K,KT,size,dd1,dd2,mu)

x=reshape(x,[size,size]);

HE1=mu.*K(x);

HE1=KT(HE1);

HE2=DhT(dd1.*Dh(x))+DvT(dd2.*Dv(x));
HE=HE1+HE2;
HE=HE(:);
end
