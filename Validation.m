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

beta_0=0;
maxit=20;
tol=10^(-6);
type=1;

[~,~,betavec1]=gn(DTD_FT,H_FT,HTH_FT,b_folder1,b_folder2,b_folder3,xF_folder1,xF_folder2,...
    xF_folder3,beta_0,maxit,tol,0.2,0.2,1,type,sigma1,sigma2,sigma3);
type=2;
[~,~,betavec2]=gn(DTD_FT,H_FT,HTH_FT,b_folder1,b_folder2,b_folder3,xF_folder1,xF_folder2,...
    xF_folder3,beta_0,maxit,tol,0.2,0.2,1,type,sigma1,sigma2,sigma3);
type=3;
[~,~,betavec3]=gn(DTD_FT,H_FT,HTH_FT,b_folder1,b_folder2,b_folder3,xF_folder1,xF_folder2,...
    xF_folder3,beta_0,maxit,tol,0.2,0.2,1,type,sigma1,sigma2,sigma3);


xF_folder1='Validation+learning\patch_validation\Sigma1';
xF_folder2='Validation+learning\patch_validation\Sigma2';
xF_folder3='Validation+learning\patch_validation\Sigma3';

b_folder1='Validation+learning\corrupted_patch_validation\Sigma1';
b_folder2='Validation+learning\corrupted_patch_validation\Sigma2';
b_folder3='Validation+learning\corrupted_patch_validation\Sigma3';

xF1 = dir(fullfile(xF_folder1,'*.jpg'));
xF2 = dir(fullfile(xF_folder2,'*.jpg'));
xF3 = dir(fullfile(xF_folder3,'*.jpg'));

b1 = dir(fullfile(b_folder1,'*.jpg'));
b2 = dir(fullfile(b_folder2,'*.jpg'));
b3= dir(fullfile(b_folder3,'*.jpg'));

numell=numel(xF1)+numel(xF2)+numel(xF3)


for j=1:length(betavec1)
    [~,f,~,p,s]=Jacfun(DTD_FT,H_FT,HTH_FT,betavec1(j),b_folder1,b_folder2,...
       b_folder3,xF_folder1,xF_folder2,xF_folder3,1,sigma1,sigma2,sigma3);
   a(j)=1/(2*numell)*norm(f)^2;
end
figure
subplot(3,1,1)
plot(1:length(betavec1),a,'-o','Linewidth',1)
xlabel('i')
ylabel('\epsilon(\mu_i)')
title('Validation set for \epsilon(\mu)') 

for j=1:length(betavec2)
    [~,f]=Jacfun(DTD_FT,H_FT,HTH_FT,betavec2(j),b_folder1,b_folder2,...
       b_folder3,xF_folder1,xF_folder2,xF_folder3,2,sigma1,sigma2,sigma3);
   b(j)=1/(numell)*norm(f)^2;
end
subplot(3,1,2)
plot(1:length(betavec2),b,'-o','Linewidth',1)
xlabel('i')
ylabel('g(\mu_i)')
title('Validation set for g(\mu)') 


for j=1:length(betavec3)
    [~,f]=Jacfun(DTD_FT,H_FT,HTH_FT,betavec3(j),b_folder1,b_folder2,...
       b_folder3,xF_folder1,xF_folder2,xF_folder3,3,sigma1,sigma2,sigma3);
   c(j)=1/(numell)*norm(f)^2;
end
subplot(3,1,3)
plot(1:length(betavec3),c,'-o','Linewidth',1)
xlabel('i')
ylabel('W(\mu_i)')
title('Validation set for W(\mu)') 
