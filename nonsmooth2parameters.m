clear all
close all
clc

xF=imread("input\.png");

xF=imresize_old(xF,[150,150]);

xF=im2double(im2gray(xF));


[n,m]=size(xF);

n=min(n,m);

xF=xF(1:n,1:n);

m=5;% support PSF           
[PSFtilde,~]=psfGauss([m,m],3);

H_FT=psf2otf(PSFtilde,[n,n]);

Dh_FT=psf2otf([1,-1],[n,n]);

Dv_FT=psf2otf([1;-1],[n,n]);

epsi=10^(-3);

b = real(ifft2(H_FT.*fft2(xF)));

M = roipoly(xF);

[k]=find(M==1);

[l]=find(M==0);
%%
randn('seed',17)

sigma1=0.05;

sigma2=0.005;

A=randn(n);

noise1=sigma1*A.*M;

noise2=sigma2*A.*(ones(n)-M);

bb=b+noise2+noise1;

x0=bb;

%load("datapepperscomplete(grigliapiùfine).mat")

x0=bb;

maxit1=500;

maxit2=50;

alpha=1;

tol=10^(-5);

maxitpcg=100000;
tolpcg=10^(-3);
tol_r=10^(-6);
toln=10^(-7);
rel_res_sf=3;

sigma_mat=(1/sigma1)*M+(1/sigma2)*(ones(size(M))-M)

%%

beta_0=[3,3];

[betaa,fk1,x1,time1,PSNR1,SSIM1,ff]=tikgnSTVimplicit2d(3,maxit1,70,x0,alpha,beta_0,bb,...
H_FT,tol,xF,maxitpcg,tolpcg,tol_r,toln,sigma1,sigma2,sigma_mat,rel_res_sf,M,k,l);

mu_mse=exp(betaa);

%%


[betaa,fk2,x2,time2,PSNR2,SSIM2]=gnSTVimplicit2d(2,maxit1,maxit2,x0,alpha,beta_0,bb,...
    epsi,H_FT,tol,xF,maxitpcg,tolpcg,tol_r,toln,sigma1,sigma2,sigma_mat,rel_res_sf,M,k,l);

mu_g=exp(betaa);

[betaa,fk3,x3,time3,PSNR3,SSIM3]=gnSTVimplicit2d(3,maxit1,maxit2,x0,alpha,beta_0,bb,...
    epsi,H_FT,tol,xF,maxitpcg,tolpcg,tol_r,toln,sigma1,sigma2,sigma_mat,rel_res_sf,M,k,l);

mu_W=exp(betaa);

%%
load("3lossbilivelshepplogan.mat")
figure
subplot(3,1,1)
plot(1/2*fk1,'Linewidth',1)
xlabel("i")
ylabel("MSE(\mu_i)")
title("GN with MSE loss implicit differentiation iteration")

subplot(3,1,2)
plot(1/2*fk2,'Linewidth',1)
xlabel("i")
ylabel("g(\mu_i)")
title("GN with gaussianity loss implicit differentiation iteration")

subplot(3,1,3)
plot(fk3,'Linewidth',1)
xlabel("i")
ylabel("W(\mu_i)")
title("GN with whiteness loss implicit differentiation iteration")

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
title("Restored image with MSE loss")

subplot(1,3,2)
imshow2(x2)
title("Restored image with gaussianity loss")

subplot(1,3,3)
imshow2(x3)
title("Restored image with whiteness loss")


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