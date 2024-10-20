function [grad,fun,ss,relres]=nonsmoothgradientimplicit(type,epsi,H_FT,HTH_FT,beta,...
    xstar,xF,bb,maxit,tol,sigma,ss0)
%ciò che andiamo a fare e andare a risolvere il sistema lineare (3.18).
%N.B. nella tesi H_E sta per hessian, la E sta per la seconda lettera di
%hessiana.
mu=exp(beta); 

muHTH_FT=mu*HTH_FT;

%minustwomuHTb=mu*minustwoHTb;

RHS=(-mu)*real(ifft2 ( conj(H_FT).*(H_FT.*fft2(xstar) - fft2(bb) ) ) );
%RHS rappresenta il termine noto del sistema lineare H_E^{-1}*RHS, ovvero
%la derivata (col meno davanti) del gradiente rispetto a e^\beta
%moltiplicata per la derivata di e^\beta rispetto a \beta.

RHS=RHS(:);

Dhu=Dh(xstar); %D_h e D_v sono nested function inserite in fondo al codice.
Dvu=Dv(xstar);

[dd1]=huber_2(Dhu,epsi);
[dd2]=huber_2(Dvu,epsi);

n=length(H_FT); %N.B. nel caso di una matrice mxn, length restituisce la dimensione
% più grande tra m e n. In tal caso abbiamo un'immagine quadrata.


[ss,~,relres]=minres(@(x)hessian(x,muHTH_FT,dd1,dd2,n),RHS,tol,maxit,[],[],ss0);
%risolviamo il sistema lineare 3.18 utilizzando MINRES. N.B:
% 1) dobbiamo usare il MINRES e non il CG perchè essendo la matrice
% semidefinita positiva e non (per forza) definita positiva, non si può
% usare il conjugate gradient.
% 2) usiamo pure qui una sorta di warm start, come si vede in G-N ss0 (mi scuso se alcune 
% variabili hanno un nome randomico, ma non sapevo come nominarle)
% viene inizializzata a [], e poi viene riutilizzata ogni volta come punto
% di partenza nel minres. 

s=reshape(ss,[n,n]);

funn=xstar(:)-xF(:);
if type==1
    grad=s(:); % il gradiente della derivata interna all'MSE è sempre la derivata
    %di x*(\beta) rispetto a \beta.
    fun=funn; 
elseif type==2
    grad=real(ifft2(H_FT.*fft2(s))); % molto simile al caso Tikhonov, questo
    % è la derivata rispetto a \beta del residuo r(\beta) 
    Res=real(ifft2(H_FT.*fft2(xstar)))-bb; % calcoliamo il residuo r(\beta)
    grad=grad(:);
    Res=Res(:);
    grad=2*grad'*Res; %derivata rispetto a \beta di (||r(\beta)||_2^2-\sigma^2*n^2)
    fun=norm(Res)^2-(sigma^2)*n^2; % (||r(\beta)||_2^2-\sigma^2*n^2)
elseif type==3
    gder=real(ifft2(H_FT.*fft2(s)));
    Res=real(ifft2(H_FT.*fft2(xstar)))-bb;
    ccorrelation1=real(ifft2(fft2(gder).*conj(fft2(Res))))+real(ifft2(fft2(Res).*conj(fft2(gder))));
    %derivata rispetto a \beta dell'autocorrelazione circolante r(\beta) o
    %r(\beta)
    ccorrelation2=real(ifft2(fft2(Res).*conj(fft2(Res)))); %autocorrelazione
    gder=gder(:);
    Res=Res(:);
    Resnorm=norm(Res,2);
    resgrad=2*gder'*Res;
    fun=ccorrelation2(:)/Resnorm^2;
    grad=(ccorrelation1(:)*Resnorm^2-ccorrelation2(:)*resgrad)/Resnorm^4;
    %derivata rispetto a \beta di (r(\beta) o r(\beta))/||r(\beta)||_2^2.
end
end

function HE = hessian(x,muHTH_FT,dd1,dd2,n)
x=reshape(x,[n,n]); %so che può sembrare inutile, ma serve fare un reshape,
%visto che minres considera x come un vettore.
HE=real(ifft2(muHTH_FT.*fft2(x)))+DhT(dd1.*Dh(x))+DvT(dd2.*Dv(x)); %calcoliamo
% la matrice hessiana HE.
HE=HE(:); %restituiamo al minres l'hessiana vettorizzata.
end

%nested function per il calcolo di D_hx, D_vx , D_h^Tx, D_v^Tx.

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
