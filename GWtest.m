clear all
close all
clc

% In questo file ho effettuato tutti i test per il bilivello per Tikhonov
% come lower level problem. 

xF=imread("input/21_peppers256.png"); %immagine originale
xF=imresize(xF,[150,150]); %facciamo un resize dell'immagine
xF=im2double(im2gray(xF)); % rendiamo l'immagine double e in scala di grigi
[n,m]=size(xF);
n=min(n,m);
xF=xF(1:n,1:n); %rendiamo l'immagine quadrata

PSFtilde=psfGauss([5,5],3); 

H_FT=psf2otf(PSFtilde,[n,n]); % calcoliamo la fft2 della matrice di blur H

b = real(ifft2(H_FT.*fft2(xF))); % blurriamo l'immagine

randn('seed',17)
sigma=0.01; %impostiamo la standard deviation del rumore
noise = sigma*randn(n,n); % creiamo un rumore bianco gaussiano
bb=b+noise;

Dh_FT=psf2otf([1,-1],size(bb)); % fft2 di D_h
Dv_FT = psf2otf([1;-1],size(bb)); % fft2 di D_v

DhT_FT=conj(Dh_FT); % fft2 di D_h^T
DvT_FT=conj(Dv_FT); % fft2 di D_v^T
DTD_FT=DhT_FT .* Dh_FT + DvT_FT .* Dv_FT;
HTH_FT=conj(H_FT).*H_FT;

bbhat=fft2(bb);
HTbb_FT=conj(H_FT).*bbhat;
mu_min = 0.005;
mu_max =3000;
mu = logspace(log10(mu_min), log10(mu_max),3000);
time_grid=0;
err=zeros(length(mu),1);
W=err;
fg=err;
peaksnr=err;
structsim=err;
for j=1:length(mu) % calcoliamo le tre loss attraverso grid-search.
    tstart=tic;
    sol_FT = HTbb_FT./(HTH_FT + DTD_FT/mu(j));
    xFF = real(ifft2(sol_FT));
    peaksnr(j)=psnr(xF,xFF);
    structsim(j)=ssim(xF,xFF);

    err(j) = (1/2)*norm(xFF(:)-xF(:))^2; %MSE
    
    fg(j)=gaussianity(H_FT,bb,HTbb_FT,HTH_FT,DTD_FT,mu(j),sigma,n); %Gaussianity
    W(j)=GRWP(H_FT,bb,HTbb_FT,HTH_FT,DTD_FT,mu(j)); % Residual Whiteness Principle (la G iniziale...
    % starebbe per Generalized (non so perchè io l'abbia chiamata così).
    a=toc(tstart);
    time_grid=time_grid+a; %tempo utilizzato per grid-search.
    
end

[errormin,l]=min(err); 
mu_err=mu(l);

[~,i]=min(fg);

mu_gauss=mu(i);

[~,j]=min(W);

mu_W=mu(j);

tol=10^-6;
maxit=100;
beta_0=2;

[beta,fk1,time1,x1]=gn1D(DTD_FT,H_FT,HTH_FT,bbhat,xF,beta_0,maxit,tol,1,sigma,n,bb);
% troviamo il punto di minimo dell'MSE attraverso bilivello

[betaa,fk2,time2,x2]=gn1D(DTD_FT,H_FT,HTH_FT,bbhat,xF,beta_0,maxit,tol,2,sigma,n,bb);
% troviamo il punto di minimo della gaussianity attraverso bilivello

g_GN=1/2*fk2(end); % valore della gaussianity nel punto di minimo

er_GN=1/2*fk1(end); % valore dell'MSE nel punto di minimo

[beta_GNW,fk3,time3,x3]=gn1D(DTD_FT,H_FT,HTH_FT,bbhat,xF,beta_0,maxit,tol,3,sigma,n,bb);
% troviamo il punto di minimo della whiteness attraverso bilivello

%%

load("Grid for one parameter\Tikhonov\PEPPERSSIGMA=0.01.mat")
% se caricate qui direttamente uno dei quattro file potete osservare
% direttamente la correttezza dell'algoritmo. (potrebbe non corrispondere
% al 100 % con i test nella tesi, visto che qui avevo usato imresize_old,
% lì imresize).


%GNW=GRWP(H_FT,bb,HTbb_FT,HTH_FT,DTD_FT,exp(beta_GNW));

GNW=fk3(end);

peaksnr_opt=psnr(xF,x1); %psnr dell'immagine restaurata ottenuta tramite MSE
structsim_opt=ssim(xF,x1); %ssim dell'immagine restaurata ottenuta tramite MSE

peaksnr_G=psnr(xF,x2); %psnr dell'immagine restaurata ottenuta tramite gaussianity
structsim_G=ssim(xF,x2); %ssim dell'immagine restaurata ottenuta tramite gaussianity

peaksnr_W=psnr(xF,x3); %psnr dell'immagine restaurata ottenuta tramite whiteness
structsim_W=ssim(xF,x3); %ssim dell'immagine restaurata ottenuta tramite whiteness

% in questa figure visualizziamo l'immagine originale e l'immagine
% osservata
figure
subplot(1,2,1)
imshow2(xF)
title("True image")
subplot(1,2,2)
imshow2(bb)
title("Observed image")
%in questa figure osserviamo le immagine x1,x2,x3, rispettivamente
%ricostruite con MSE, Gaussianity e Whiteness bilivello
figure,
subplot(1,3,1)
imshow2(x1)
title("Optimal restoration by bilevel MSE")
subplot(1,3,2)
imshow2(x2)
title("Optimal restoration by bilevel Gaussianity")
subplot(1,3,3)
imshow2(x3)
title("Optimal restoration by bilevel Whiteness")

% psnr e ssim al variare del valore di mu, con il valore per le immagini
% ricostruite con le tre loss
figure

subplot(2,1,1)
loglog(mu,peaksnr,'Linewidth',1)

hold on

loglog(mu_err,peaksnr_opt,'o','Linewidth',1)
hold on
loglog(mu_gauss,peaksnr_G,'o','Linewidth',1)
hold on
loglog(mu_W,peaksnr_W,'o','Linewidth',1)

legend('PSNR(\beta)','PSNR (MSE)','PSNR (Gaussianity)','PSNR (Whiteness)','Location','southwest')

ylabel('PSNR(\beta)')
xlabel('e^\beta')
ylim([0,max(peaksnr)+2])

title('Function PSNR(\beta)')

subplot(2,1,2)

loglog(mu,structsim,'Linewidth',1)

hold on

loglog(mu_err,structsim_opt,'o','Linewidth',1)
hold on
loglog(mu_gauss,structsim_G,'o','Linewidth',1)
hold on
loglog(mu_W,structsim_W,'o','Linewidth',1)

legend('SSIM(\beta)','SSIM (MSE)','SSIM (Gaussianity)','SSIM (Whiteness)','Location','southwest')

ylabel('SSIM(\beta)')
xlabel('e^\beta')

title('Function SSIM(\beta)')
ylim([0,max(structsim)+0.1])

[m,k]=min(err);

% plot delle tre loss con il grafico lungo le iterazioni. Verifichiamo che
% il punto di minimo ottenuto con bilivello coincida con quello ottenuto
% con grid-search.
figure
subplot(3,1,1)
loglog(mu,err,'Linewidth',1)
hold on
loglog(exp(beta),er_GN,'o','Linewidth',1)
hold on
loglog(mu(k),m,'*','Linewidth',1)
xlabel('e^\beta')
ylabel('MSE(\beta)')
legend('MSE(\beta)','MSE(\beta*) (Gauss-Newton)','MSE(\beta*) (grid-search)')

title('Function MSE(\beta)')

subplot(3,1,2)


loglog(mu,fg,'Linewidth',1)
hold on
loglog(exp(betaa),fk2(end),'o','Linewidth',1)
hold on
loglog(mu(i),min(fg),'*','Linewidth',1)


legend('g(\beta)','g(\beta*) (Gauss-Newton)','g(\beta*) (grid-search)')
xlabel('e^\beta')
ylabel('g(\beta)')
title("Function g(\beta)")

subplot(3,1,3)


loglog(mu,W,'Linewidth',1)

hold on

loglog(exp(beta_GNW),GNW,'o','Linewidth',1)

hold on

loglog(mu_W,min(W),'*','Linewidth',1)
legend('W(\beta)','W(\beta*) (Gauss-Newton)','W(\beta*) (grid-search)')
title("Function W(\beta)")

xlabel('e^\beta')
ylabel('W(\beta)')

% verifichiamo che le tre loss decrescano lungo le iterazioni di
% Gauss-Newton
figure
subplot(3,1,1)
plot(1/2*fk1,'Linewidth',1)
xlabel("i")
ylabel("MSE(\beta_i)")
title("MSE loss decays along iterations")

subplot(3,1,2)
plot(1/2*fk2,'Linewidth',1)
xlabel("i")
ylabel("g(\beta_i)")
title("Gaussianity loss decays along iterations")

subplot(3,1,3)
plot(fk3,'Linewidth',1)
xlabel("i")
ylabel("W(\beta_i)")
title("Whiteness loss decays along iterations")

%save("Grid for one parameter\Tikhonov\PEPPERSSIGMA=0.01.mat")

