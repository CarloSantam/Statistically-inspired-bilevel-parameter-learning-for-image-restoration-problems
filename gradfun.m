function [grad,f,p,s,Rf]=gradfun(DTD_FT,H_FT,HTH_FT,beta,bbhat,xF)
R_FT=1./(DTD_FT+exp(beta).*HTH_FT);
RR_FT=exp(beta)*R_FT.*conj(H_FT);
Rf=real(ifft2(RR_FT.*bbhat)); % calcoliamo x*(e^(\beta)) per...
% Tikhonov  
RRf1=real(ifft2((R_FT.*HTH_FT.*R_FT.*conj(H_FT)).*bbhat)); %calcoliamo la prima parte della derivata...
% di x*(e^\beta) rispetto a \beta (formula (3.5)-(3.6), del terzo capitolo in tesi)....
% (per whiteness e gaussianity ometto i commenti, visto che sono analoghi per questa prima parte)
RRf2=real(ifft2((R_FT.*conj(H_FT).*bbhat))); % calcoliamo la seconda parte del gradiente...
%(stessa formula citata sopra)
grad=-exp(2*beta)*RRf1(:)+exp(beta)*RRf2(:); %calcoliamo il gradiente (stessa formula).
f=Rf(:)-xF(:); %calcoliamo x^(\mu)-x_true (ricordiamo che per Gauss-Newton sono necessari solamente...
%la jacobiana rispetto al parametro \mu e l'argomento all'interno della
%della norma. 
p=psnr(xF,Rf); %il calcolo del ssim e psnr compare direttamente nel caso con più immagini
s=ssim(xF,Rf);
end
