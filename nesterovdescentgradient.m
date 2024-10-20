function [x,ff]=nesterovdescentgradient(maxit,x0,mu,bb,epsi,H_FT,tol)
%x=x0;
x1=x0; 
y=x0;
t=1;

alpha=1/((mu)+12/epsi); %fissiamo il passo di discesa a 1/L, dove L è la
%costante di Lipschitz del gradiente che ho pre-calcolato (nella tesi,
%secondo capitolo, Teorema 2.1)
[f2]=fun(mu,bb,epsi,x0,H_FT); %inizializziamo la funzione f, che indica la
%decrescita della funzione \mu/2*||Hx-b||_2^2+sum_{i} n_\epsi(D_hx)_i+...
% sum_{i} n_\epsi(D_vx)_i, fun è una nested function in fondo al codice
ff=nan(maxit,1); % svolgiamo un pre-allocamento di ff.
ff(1)=f2;

%Le iterazioni seguenti, sono quelle di Nesterov standard

for i=1:maxit
    
    [g]=gradient(H_FT,y,bb,epsi,mu);
    x=y-alpha*g;

    tnew=(1+sqrt(1+4*t^2))/2;
    y=x+((t-1)/tnew)*(x-x1);

    x1=x;

    t=tnew;

    [f2]=fun(mu,bb,epsi,x,H_FT);

    ff(i+1)=f2;

    if norm(ff(i+1)-ff(i))/norm(ff(i))<tol
        break
    end

end
end

function [f]=fun(mu,bb,epsi,x,H_FT) %Nested function per calcolare ...
% \mu/2*||Hx-b||_2^2+sum_{i} n_\epsi(D_hx)_i+sum_{i} n_\epsi(D_vx)_i

z=real(ifft2(H_FT.*fft2(x)))-bb;
Dhx=[ x(:,2:end) - x(:,1:(end-1)) , x(:,1) - x(:,end) ];
Dvx=[ x(2:end,:) - x(1:(end-1),:) ; x(1,:) - x(end,:) ];

%{
if 0
[L1,dL1]=huber(Dhx(:),epsi);


[L2,dL2]=huber(Dvx(:),epsi);

f=mu/2*norm(z,'fro')^2+sum(L1)+sum(L2);

dL1=reshape(dL1,[length(Dhx),length(Dhx)]);
dL2=reshape(dL2,[length(Dvx),length(Dvx)]);

else
%}

    [L1]=huber_0(Dhx,epsi);

    [L2]=huber_0(Dvx,epsi);

    f=(mu/2)*norm(z,'fro')^2+sum(L1(:))+sum(L2(:));

end

function g=gradient(H_FT,x,bb,epsi,mu) %calcoliamo il gradiente (la cui...
% form è stata calcolata esplicitamente in Theorem 2.1 del capitolo 2)

z=real(ifft2(H_FT.*fft2(x)))-bb;

Dhx=[ x(:,2:end) - x(:,1:(end-1)) , x(:,1) - x(:,end) ];
Dvx=[ x(2:end,:) - x(1:(end-1),:) ; x(1,:) - x(end,:) ];
%DhTx = [ x(:,end) - x(:,1) , -diff(x,1,2) ];
%DvTx = [ x(end,:) - x(1,:) ; -diff(x,1,1) ];

[dL1]=huber_1(Dhx,epsi);

[dL2]=huber_1(Dvx,epsi);

g1=mu*real(ifft2(conj(H_FT).*fft2(z))); %mu*H^T(Hx-b)

g2=[ dL1(:,end) - dL1(:,1) , -diff(dL1,1,2) ];

g3=[ dL2(end,:) - dL2(1,:) ; -diff(dL2,1,1) ];


g=g1+g2+g3;

end

