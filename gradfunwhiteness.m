function [grad,f,Rf]=gradfunwhiteness(DTD_FT,H_FT,HTH_FT,beta,bbhat,bb)
R_FT=1./(DTD_FT+exp(beta).*HTH_FT);
RR_FT=exp(beta)*R_FT.*conj(H_FT);
Rf=real(ifft2(RR_FT.*bbhat)); 
RRf1=real(ifft2((R_FT.*HTH_FT.*R_FT.*conj(H_FT)).*bbhat));
RRf2=real(ifft2((R_FT.*conj(H_FT).*bbhat)));
grad=-exp(2*beta)*RRf1+exp(beta)*RRf2;

gder=real(ifft2(H_FT.*fft2(grad)));
Res=real(ifft2(H_FT.*fft2(Rf)))-bb;


ccorrelation1=real(ifft2(fft2(gder).*conj(fft2(Res))))+real(ifft2(fft2(Res).*conj(fft2(gder))));
%calcoliamo la circulant cross correlation (nei commenti la scrivo come
%'o') r(\mu)o(derivata di r(\mu) rispetto a \mu) e (derivata di r(\mu)
%rispetto a \mu) o r(\mu))
ccorrelation2=real(ifft2(fft2(Res).*conj(fft2(Res)))); %calcoliamo r(\mu) o r(\mu)

gder=gder(:);
Res=Res(:);
Resnorm=norm(Res,2);
resgrad=2*gder'*Res; %calcoliamo la derivata della norma euclidea elevata al quadrato del residuo )
f=ccorrelation2(:)/Resnorm^2; %calcoliamo l'argomento della norma nella funzione whiteness
grad=(ccorrelation1(:)*Resnorm^2-ccorrelation2(:)*resgrad)/Resnorm^4; %caloliamo...
%la derivata-gradiente rispetto a \mu dell'argomento nella whiteness.
end
