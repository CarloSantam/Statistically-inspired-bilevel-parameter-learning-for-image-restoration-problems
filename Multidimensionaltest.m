clear all
close all
clc

sigma1=0.001;
sigma2=0.01;
sigma3=0.1;

xF_folder1='Validation+learning\patch_learning\Sigma1';
xF_folder2='Validation+learning\patch_learning\Sigma2';
xF_folder3='Validation+learning\patch_learning\Sigma3';

b_folder1='Validation+learning\corrupted_patch_learning\Sigma1';
b_folder2='Validation+learning\corrupted_patch_learning\Sigma2';
b_folder3='Validation+learning\corrupted_patch_learning\Sigma3';

xF1 = dir(fullfile(xF_folder1,'*.jpg'));
xF2 = dir(fullfile(xF_folder2,'*.jpg'));
xF3 = dir(fullfile(xF_folder3,'*.jpg'));

b1 = dir(fullfile(b_folder1,'*.jpg'));
b2 = dir(fullfile(b_folder2,'*.jpg'));
b3= dir(fullfile(b_folder3,'*.jpg'));


numel(b1)
n=100;
m=11;% support PSF           
[PSFtilde,~]=psfGauss([m,m],4);
H_FT=psf2otf(PSFtilde,[n,n]);
Dh_FT=psf2otf([1,-1],[n,n]);
Dv_FT = psf2otf([1;-1],[n,n]);
DhT_FT=conj(Dh_FT);
DvT_FT=conj(Dv_FT);
DTD_FT=DhT_FT .* Dh_FT + DvT_FT .* Dv_FT;
HTH_FT=conj(H_FT).*H_FT;

mu_min=0.005;
mu_max=1e+08;

numell=numel(xF1)+numel(xF2)+numel(xF3)

mu=logspace(log10(mu_min),log10(mu_max),100);

for j=1:length(mu)
   [~,f,~,p,s]=Jacfun(DTD_FT,H_FT,HTH_FT,log(mu(j)),b_folder1,b_folder2,...
       b_folder3,xF_folder1,xF_folder2,xF_folder3,1,sigma1,sigma2,sigma3);
   a(j)=(1/(2*numell))*norm(f,2)^2;
   pp(j)=1/numell*sum(p);
   ss(j)=1/numell*sum(s);
   
   [~,f,~]=Jacfun(DTD_FT,H_FT,HTH_FT,log(mu(j)),b_folder1,b_folder2,...
       b_folder3,xF_folder1,xF_folder2,xF_folder3,2,sigma1,sigma2,sigma3);
   b(j)=1/(2*numell)*norm(f)^2;
   
   [~,f,SSSS(j)]=Jacfun(DTD_FT,H_FT,HTH_FT,log(mu(j)),b_folder1,b_folder2,...
       b_folder3,xF_folder1,xF_folder2,xF_folder3,3,sigma1,sigma2,sigma3);
   c(j)=1/numell*norm(f)^2;
end
figure
subplot(3,1,1)
loglog(mu,a,'Linewidth',1)
hold on
[m,k]=min(a);
beta_0=0;
maxit=20;
tol=10^(-6);
type=1;

[beta1,S1,betavec1]=gn(DTD_FT,H_FT,HTH_FT,b_folder1,b_folder2,b_folder3,xF_folder1,xF_folder2,...
    xF_folder3,beta_0,maxit,tol,0.2,0.2,1,type,sigma1,sigma2,sigma3);

muu=exp(beta1);

[~,f_err,~]=Jacfun(DTD_FT,H_FT,HTH_FT,beta1,b_folder1,b_folder2,b_folder3,...
    xF_folder1,xF_folder2,xF_folder3,1,sigma1,sigma2,sigma3);
err_tot=1/(2*numell)*norm(f_err)^2;
loglog(muu,err_tot,'o','Linewidth',1)
hold on
loglog(mu(k),m,'*','Linewidth',1)
legend('MSE(\mu)','MSE(\mu*)(Line search Gauss-Newton)','MSE(\mu*)(grid)')

xlabel('\mu')
ylabel('$MSE(\mu)=\frac{1}{k_T}\sum_{i=1}^{k_T}\frac{1}{2}||x_i(\mu)-xi||_2^2$','interpreter', 'LaTex')

title('MSE(\mu) with 216 images')

type=2;

[m,k]=min(b);

subplot(3,1,2)


loglog(mu,b,'Linewidth',1)
hold on
[beta2,S2,betavec2]=gn(DTD_FT,H_FT,HTH_FT,b_folder1,b_folder2,b_folder3,xF_folder1,xF_folder2,...
    xF_folder3,beta_0,maxit,tol,0.2,0.2,1,type,sigma1,sigma2,sigma3);

[~,g_tot,~]=Jacfun(DTD_FT,H_FT,HTH_FT,beta2,b_folder1,b_folder2,b_folder3,...
    xF_folder1,xF_folder2,xF_folder3,type,sigma1,sigma2,sigma3);

g_tot=norm(g_tot)^2/(2*numell);

loglog(exp(beta2),g_tot,'o','Linewidth',1)

xlabel('\mu')
ylabel('$g(\mu)=\frac{1}{k_T}\sum_{i=1}^{k_T}g_i(\mu)$','interpreter', 'LaTex')

hold on

loglog(mu(k),m,'*','Linewidth',1)

title('g(\mu) with 216 images')

legend('g(\mu)','g(\mu*)(Line search Gauss Newton)','g(\mu*)(grid)')

subplot(3,1,3)
type=3;
beta_0=0;
[m,k]=min(c);
loglog(mu,c,'Linewidth',1)
hold on
[beta3,S3,betavec3]=gn(DTD_FT,H_FT,HTH_FT,b_folder1,b_folder2,b_folder3,xF_folder1,xF_folder2,...
    xF_folder3,beta_0,maxit,tol,0.2,0.2,1,type,sigma1,sigma2,sigma3);

[~,W_tot,~]=Jacfun(DTD_FT,H_FT,HTH_FT,beta3,b_folder1,b_folder2,b_folder3,...
    xF_folder1,xF_folder2,xF_folder3,type,sigma1,sigma2,sigma3);

W_tot=norm(W_tot)^2/numell;

loglog(exp(beta3),W_tot,'o','Linewidth',1)

xlabel('\mu')
ylabel('$W(\mu)=\frac{1}{k_T}\sum_{i=1}^{k_T}W_i(\mu)$','interpreter', 'LaTex')

hold on

loglog(mu(k),m,'*','Linewidth',1)

title('W(\mu) with 216 images')

legend('W(\mu)','W(\mu*)(Line search Gauss Newton)','W(\mu*)(grid)')

figure
subplot(3,1,1)

plot(1:length(S1),S1/(2*numell),'-o','Linewidth',1)
xlabel('i')
ylabel('$MSE(\beta_i+\alpha_id)$','interpreter', 'LaTex')

title('Line search Gauss Newton iteration for MSE(\mu)')

subplot(3,1,2)


plot(1:length(S2),S2/numell,'-o','Linewidth',1)
xlabel('i')
ylabel('$g(\beta_i+\alpha_id)$','interpreter', 'LaTex')
title('Line search Gauss Newton iteration for g(\mu)')


subplot(3,1,3)


plot(1:length(S3),S3/numell,'-o','Linewidth',1)
xlabel('i')
ylabel('$W(\beta_i+\alpha_id)$','interpreter', 'LaTex')
title('Line search Gauss Newton iteration for W(\mu)')


figure
subplot(2,1,1)


loglog(mu,pp,'Linewidth',1)

hold on
[~,~,~,p,s1]=Jacfun(DTD_FT,H_FT,HTH_FT,beta1,b_folder1,b_folder2,...
       b_folder3,xF_folder1,xF_folder2,xF_folder3,1,sigma1,sigma2,sigma3);
   
   pp=1/numell*sum(p);
loglog(exp(beta1),pp,'o','Linewidth',1)

hold on
[~,~,~,p,s2]=Jacfun(DTD_FT,H_FT,HTH_FT,beta2,b_folder1,b_folder2,...
       b_folder3,xF_folder1,xF_folder2,xF_folder3,1,sigma1,sigma2,sigma3);
   
   pp=1/numell*sum(p);
loglog(exp(beta2),pp,'o','Linewidth',1)

hold on
[~,~,~,p,s3]=Jacfun(DTD_FT,H_FT,HTH_FT,beta3,b_folder1,b_folder2,...
       b_folder3,xF_folder1,xF_folder2,xF_folder3,1,sigma1,sigma2,sigma3);
   
   pp=1/numell*sum(p);
loglog(exp(beta3),pp,'o','Linewidth',1)
legend('PSNR','PSNR(MSE)','PSNR(gaussianity)','PSNR(whiteness)','location','southwest')
xlabel('\mu')
ylabel('$PSNR(\mu)=\frac{1}{k_T}\sum_{i=1}^{m}PSNR(x_i(\mu),x_i)$','interpreter','LaTex')

subplot(2,1,2)


loglog(mu,ss,'Linewidth',1)
hold on
sss=1/numell*sum(s1);
loglog(exp(beta1),sss,'o','Linewidth',1)
hold on
sss=1/numell*sum(s2);
loglog(exp(beta2),sss,'o','Linewidth',1)
hold on
sss=1/numell*sum(s3);
loglog(exp(beta3),sss,'o','Linewidth',1)

legend('SSIM','SSIM(MSE)','SSIM(gaussianity)','SSIM(whiteness)','location','southwest')
xlabel('\mu')
ylabel('$SSIM(\mu)=\frac{1}{k_T}\sum_{i=1}^{m}SSIM(x_i(\mu),x_i)$','interpreter','LaTex')

