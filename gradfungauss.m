function [grad,f,Rf]=gradfungauss(DTD_FT,H_FT,HTH_FT,beta,bbhat,n,sigma)
R_FT=1./(DTD_FT+exp(beta).*HTH_FT);
RR_FT=exp(beta)*R_FT.*conj(H_FT);
Rf=real(ifft2(RR_FT.*bbhat));% calcoliamo x*(e^(\beta)) per...
% Tikhonov  
RRf1=real(ifft2((R_FT.*HTH_FT.*R_FT.*conj(H_FT)).*bbhat));%calcoliamo la prima parte della derivata...
% di x*(e^\beta) rispetto a \beta (formula (3.5)-(3.6), del terzo capitolo in tesi)....
% (per whiteness e gaussianity ometto i commenti, visto che sono analoghi per questa prima parte)
RRf2=real(ifft2((R_FT.*conj(H_FT).*bbhat)));% calcoliamo la seconda parte del gradiente...
%(stessa formula citata sopra)
grad=-exp(2*beta)*RRf1+exp(beta)*RRf2;%calcoliamo il gradiente (stessa formula).

gder=real(ifft2(H_FT.*fft2(grad))); % calcoliamo il gradiente del residuo H*(derivata di x*(\beta)...
%rispetto a \beta)
Res=real(ifft2(H_FT.*fft2(Rf)))-ifft2(bbhat); % calcoliamo il residuo Hx*-b.
gder=gder(:);
Res=Res(:);

grad=2*gder'*Res; %calcoliamo la derivata di (||r(\mu)||_2^2-\sigma^2*n^2),
%dove \sigma è la standard deviation del rumore e n è la dimensione
%dell'immagine (che assumiamo essere quadrata nxn).
f=norm(Res)^2-(sigma^2)*n^2; %calcoliamo (||r(\mu)||_2^2-\sigma^2*n^2).

end