clear all
close all
clc

% questo test era il famoso "nomedadecidere" :)

xF=imread("input\43_man256.png");
xF=imresize_old(xF,[150,150]);
xF=im2double(im2gray(xF));

[n,m]=size(xF);
n=min(n,m);
xF=xF(1:n,1:n);

m=3;% support PSF 
[PSFtilde,center]=psfGauss([m,m],3);

H_FT=psf2otf(PSFtilde,[n,n]);

b = real(ifft2(H_FT.*fft2(xF)));

randn('seed',17)
sigma=0.01;
noise = sigma*randn(n,n);
bb=b+noise;

%Dh_FT=psf2otf([1,-1],size(bb));
%Dv_FT = psf2otf([1;-1],size(bb));
%rng('default')

maxit=500;
alpha_0=1;
epsi=10^(-3); %impostiamo epsilon di huber
toln=10^(-7); %impostiamo la tolleranza per Nesterov

mu_min=0.05;

mu_max=3000;

mu=logspace(log10(mu_min),log10(mu_max),3000); %impostiamo la grigliatura
%di mu 

time_grid=0;
err=zeros(length(mu),1);
W=err;
g=err;
peak_snr=err;
s_signal=err;
t=0;
xx=bb;
for i=1:length(mu)
    tstart=tic;
    [xx,ff]=nesterovdescentgradient(maxit,xx,mu(i),bb,epsi,H_FT,toln);
    ff=rmmissing(ff)
    ff(end)-min(ff)
    %calcoliamo xx, ovvero x*(\mu)
    r=xx-xF;
    err(i)=1/2*norm(r,'fro')^2; %err corrisponde all'MSE
    
    HxFF=real(ifft2(H_FT.*fft2(xx)));
    ResF=HxFF-bb;
    g(i)=((norm(ResF(:),2).^2-sigma^2*(n^2))^2)/2; %calcoliamo la gaussianity
    
    Resnorm=norm(ResF(:)); 
    ccorrelation2=real(ifft2(fft2(ResF).*conj(fft2(ResF)))); %calcoliamo la correlazione del residuo
    f=ccorrelation2(:)/Resnorm^2; % questa è la normalized autocorrelation
    W(i)=norm(f)^2; %calcoliamo la Whiteness 
    
    peak_snr(i)=psnr(xx,xF); %psnr al variare di mu
    s_signal(i)=ssim(xx,xF);  %ssim al variare di mu
    
    time_grid=time_grid+toc(tstart); %tempo cumulativo

    t=t+1
end

figure
plot(ff,'Linewidth',1)

xlabel("$i$",'Interpreter','latex')
ylabel("$\mathcal{J}(x_i;\beta)$",'interpreter','latex')

[m,j]=min(err);
[m_G,k]=min(g);
[m_W,l]=min(W);

mu_msegrid=mu(j); %minimo dell'MSE su griglia
mu_ggrid=mu(k); %minimo della gaussianity su griglia
mu_wgrid=mu(l); %minimo della whiteness su griglia

alpha=1;
beta_0=2; %punto iniziale per Gauss-Newton
maxitminres=100000; %numero di iterazioni massimo per minres. Ricordo che usiamo minres perchè non
% abbiamo alcuna garanzia che la matrice hessiana sia definita positiva, specie non
% essendo certi che il punto di minimo del lower level sia unico. E' però semidefinita
% positiva, quindi l'algoritmo minres è perfetto a tal fine.
tolminres=10^(-1); %sempre "sbagliato"
tol=10^(-3); % questa è la tolleranza tale per cui se la direzione di discesa  d
% tale che x=x+d, ha norma ||d||_2<tol, l'algoritmo si blocca. Utile specie
% per gaussianity
tol_r=10^(-5); % questa è la tolleranza relativa.
maxit1=500; %numero di iterazioni per nesterov
maxit2=60; %numero di iterazioni per Gauss-Newton
x0=bb; %inizializzazione per Nesterov (poi, come si vedrà in G-N, si userà 
%la warm start).


toln=10^(-7); %tolleranza di Nesterov
[betaa,fk1,x1,time1,PSNR1,SSIM1]=gnSTVimplicit(1,maxit1,maxit2,x0,alpha,beta_0,bb,epsi,H_FT,tol,xF,...
maxitminres,tolminres,sigma,tol_r,toln,2); %troviamo il punto di minimo con Gauss-Newton per MSE.

%save("Test confronto\pepperssepsi=10^-3sigma=0.01.mat")


mu_mse=exp(betaa);
[betaa,fk2,x2,time2,PSNR2,SSIM2]=gnSTVimplicit(2,maxit1,maxit2,x0,alpha,beta_0,bb,epsi,H_FT,10^(-4),xF,...
maxitminres,tolminres,sigma,tol_r,toln,2);
%%troviamo il punto di minimo con Gauss-Newton per Gaussianity.

mu_g=exp(betaa);
[betaa,fk3,x3,time3,PSNR3,SSIM3]=gnSTVimplicit(3,maxit1,maxit2,x0,alpha,beta_0,bb,epsi,H_FT,tol,xF,...
maxitminres,tolminres,sigma,tol_r,toln,2);

mu_W=exp(betaa);
%%troviamo il punto di minimo con Gauss-Newton per Whiteness.

%%
%come in Tikhonov plottiamo le tre loss, e verifichiamo che effettivamente
%il punto di minimo ottenuto con grid-search sia lo stesso ottenuto con
%Gauss-Newton

load("Grid for one parameter\Total Variation\PEPPERSSIGMA=0.01.mat")


figure

subplot(3,1,1)

loglog(mu,err,'Linewidth',1)
hold on
loglog(mu_mse,1/2*fk1(end),'o','Linewidth',1)
hold on
loglog(mu_msegrid,m,'*','Linewidth',1)
xlabel("e^\beta")
ylabel("MSE(\beta)")
legend('MSE(\beta)','MSE(\beta*) (Gauss-Newton)','MSE(\beta*) (grid-search)')
title('Function MSE(\beta)')


subplot(3,1,2)

loglog(mu,g,'Linewidth',1)
hold on
loglog(mu_g,1/2*fk2(end),'o','Linewidth',1)
hold on
loglog(mu_ggrid,m_G,'*','Linewidth',1)
title('Function g(\beta)')


xlabel("e^\beta")
ylabel("g(\beta)")
legend('g(\beta)','g(\beta*) (Gauss-Newton)','g(\beta*) (grid-search)')


subplot(3,1,3)
loglog(mu,W,'Linewidth',1)
hold on
loglog(mu_W,fk3(end),'o','Linewidth',1)
hold on
loglog(mu_wgrid,m_W,'*','Linewidth',1)
xlabel("e^\beta")
ylabel("W(\beta)")
legend('W(\beta)','W(\beta*) (Gauss-Newton)','W(\beta*) (grid-search)')
title('Function W(\beta)')

%verifichiamo che le tre loss decrescano lungo le iterazioni di GN

figure
subplot(3,1,1)
plot(1/2*fk1,'Linewidth',1)
xlabel("i")
ylabel("MSE(\beta_i)")
title("MSE loss decays along  iteration")

subplot(3,1,2)
plot(1/2*fk2,'Linewidth',1)
xlabel("i")
ylabel("g(\beta_i)")
title("Gaussianity loss decays along iteration")

subplot(3,1,3)
plot(fk3,'Linewidth',1)
xlabel("i")
ylabel("W(\beta_i)")
title("Whiteness loss decays along iteration")
%osserviamo le immagini ricostruite in relazione all'immagine originale e
%quella blurrata
figure
subplot(1,2,1)
imshow2(xF)
title("Original image")

subplot(1,2,2)
imshow2(bb)
title("Observed image")

figure
subplot(1,3,1)
imshow2(x1)
title("Optimal restoration by bilevel MSE")

subplot(1,3,2)
imshow2(x2)
title("Optimal restoration by bilevel Gaussianity")

subplot(1,3,3)
imshow2(x3)
title("Optimal restoration by bilevel Whiteness")

%plottiamo il psnr e l'ssim, verificando il valore delle due misure anche
%per le tre loss
figure

subplot(2,1,1)
loglog(mu,peak_snr,'Linewidth',1)
hold on
loglog(mu_mse,psnr(x1,xF),'o','Linewidth',1)
hold on
loglog(mu_g,psnr(x2,xF),'o','Linewidth',1)
hold on
loglog(mu_W,psnr(x3,xF),'o','Linewidth',1)
title("Function PSNR(\beta)")
xlabel('\beta')
ylabel('PSNR(\beta)')
legend('PSNR','PSNR (MSE)','PSNR (Gaussianity)','PSNR (Whiteness)')
ylim([0,max(peak_snr)+2])

subplot(2,1,2)

loglog(mu,s_signal,'Linewidth',1)
hold on
loglog(mu_mse,ssim(x1,xF),'o','Linewidth',1)
hold on
loglog(mu_g,ssim(x2,xF),'o','Linewidth',1)
hold on
loglog(mu_W,ssim(x3,xF),'o','Linewidth',1)
title("Function SSIM(\beta)")
xlabel('\beta')
ylabel('SSIM(\beta)')
legend('SSIM','SSIM(MSE)','SSIM (Gaussianity)','SSIM (Whiteness)')
ylim([0,max(s_signal)+0.1])

% plottiamo il tempo cumulativo lungo le iterazioni (non presente nella
% tesi, 

figure
subplot(3,1,1)
plot(time1,'Linewidth',1)
xlabel("i")
ylabel("s")
title("Gauss-Newton Iteration-CPU time for MSE")
subplot(3,1,2)
plot(time2,'Linewidth',1)
xlabel("i")
ylabel("s")
title("Gauss-Newton Iteration-CPU time for g")
subplot(3,1,3)
plot(time3,'Linewidth',1)
xlabel("i")
ylabel("s")
title("Gauss-Newton Iteration-CPU time for W")

%plottiamo come descrescono le tre loss lungo le iterazioni in relazione
%del tempo.

figure

subplot(3,1,1)
plot(time1,fk1,'Linewidth',1)
xlabel("s")
ylabel("MSE")
title("Gauss-Newton MSE-CPU time for MSE")
subplot(3,1,2)
plot(time2,fk2,'Linewidth',1)
xlabel("s")
ylabel("g")
title("Gauss-Newton g-CPU time for g")
subplot(3,1,3)
plot(time3,fk3,'Linewidth',1)
xlabel("s")
ylabel("W")
title("Gauss-Newton W-CPU time for W")

%verifichiamo come si evolgono il PSNR e l'SSIM lungo il tempo cumulativo
%di gauss-newton (per le 3 loss differenti)

figure

subplot(3,1,1)
plot(time1,PSNR1,'Linewidth',1)
xlabel("s")
ylabel("PSNR")
title("Gauss-Newton PSNR-CPU time for MSE")
subplot(3,1,2)
plot(time2,PSNR2,'Linewidth',1)
xlabel("s")
ylabel("PSNR")
title("Gauss-Newton PSNR-CPU time for g")
subplot(3,1,3)
plot(time3,PSNR3,'Linewidth',1)
xlabel("s")
ylabel("PSNR")
title("Gauss-Newton PSNR-CPU time for W")

%SSIM per le 3 loss lungo il tempo cumulativo di Gauss-Newton

figure

subplot(3,1,1)
plot(time1,SSIM1,'Linewidth',1)
xlabel("s")
ylabel("SSIM")
title("Gauss-Newton SSIM-CPU time for MSE")
subplot(3,1,2)
plot(time2,SSIM2,'Linewidth',1)
xlabel("s")
ylabel("SSIM")
title("Gauss-Newton SSIM-CPU time for g")
subplot(3,1,3)
plot(time3,SSIM3,'Linewidth',1)
xlabel("s")
ylabel("SSIM")
title("Gauss-Newton SSIM-CPU time for W")

save("Grid for one parameter\Total Variation\MANSIGMA=0.01.mat")
