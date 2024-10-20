function [x,ff]=tiknesterovdescentgradientmp(maxit,x0,mu,bb,H_FT,tol)

%Differenze da notare rispetto al caso di prima:
% 1) il gradiente stavolta è H^T D_\mu (Hx-b)+D_h^T*D_h*x+D_v^T*D_v*x.
% 2) La costante di Lipschitz stavolta è L=max(mu(:))+8, infatti, non
% abbiamo più l'influenza di \epsilon dalla loss di Huber.
% Per il resto è identico al caso di TV_\epsilon :).

x1=x0;
y=x0;
t=1;

ff=nan(maxit,1);

%normH=sqrt(max(abs(conj(H_FT(:)).*H_FT(:))));

%normDh=sqrt(max(abs(conj(Dh_FT(:)).*Dh_FT(:))));

%normDv=sqrt(max(abs(conj(Dv_FT(:)).*Dv_FT(:))));

alpha=1/(max(mu(:))+8);

%alpha=10^(-10)

[f2]=fun(H_FT,bb,x0,mu);
ff(1)=f2;

for i=1:maxit
    
    [g]=gradient(H_FT,bb,y,mu);
     
    x=y-alpha*g;
    tnew=(1+sqrt(1+4*t^2))/2;
    y=x+((t-1)/(tnew))*(x-x1);

    x1=x;

    t=tnew;

    [f2]=fun(H_FT,bb,x,mu);

    ff(i+1)=f2;

    if abs(ff(i+1)-ff(i))/abs(ff(i+1))<tol || norm(g(:))<tol
        break
    end
    
end
end

function fun=fun(H_FT,bb,x,mu)

z=real(ifft2(H_FT.*fft2(x)))-bb;
Dhx=[ x(:,2:end) - x(:,1:(end-1)) , x(:,1) - x(:,end) ];
Dvx=[ x(2:end,:) - x(1:(end-1),:) ; x(1,:) - x(end,:) ]; 

%L1=huber_0(Dhx,epsi);

%L2=huber_0(Dvx,epsi);

A=abs(z).^2;

fun=(1/2)*sum(mu(:).*A(:))+(1/2)*norm(Dhx,'fro')^2+(1/2)*norm(Dvx,'fro')^2;

end

function g=gradient(H_FT,bb,x,mu)

z=real(ifft2(H_FT.*fft2(x)))-bb;

Dhx=[ x(:,2:end) - x(:,1:(end-1)) , x(:,1) - x(:,end) ];
Dvx=[ x(2:end,:) - x(1:(end-1),:) ; x(1,:) - x(end,:) ]; 

%dL1=huber_1(Dhx,epsi);
%dL2=huber_1(Dvx,epsi);

s=(mu).*z;
g1=real(ifft2(conj(H_FT).*fft2(s)));
 %2*mu.*H^T(Hx-b)
g2=[ Dhx(:,end) - Dhx(:,1) , -diff(Dhx,1,2) ];
g3=[ Dvx(end,:) - Dvx(1,:) ; -diff(Dvx,1,1) ];

g=g1+g2+g3;
end

