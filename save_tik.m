clear all
close all
clc

%Nulla da commentare di preciso, solo vado salvarmi tutti i risultati
%finali di tikhonov con due parametri per poi darli in pasto alla prima
%section in "plot2dWhiteness.m".

load("datapepperscomplete2.mat")
maxitpcg=100000;
tolpcg=10^(-1);
tol_r=10^(-5);
toln=10^(-7);
rel_res_sf=2;
tol=10^(-3);

beta0_tik=[0,1];


[betaa_tikerr,fktik_err,xxtik_err,time_tikerr,PSNRtik_err,SSIMtik_err,betaavecerr_tik,ff_tikerr]=tikgnSTVimplicit2d(1,500,60,x0,1,beta0_tik,bb,...
H_FT,tol,xF,maxitpcg,tolpcg,tol_r,toln,sigma1,sigma2,sigma_mat,rel_res_sf,M,k,l);

betaa_tikerr=[betaa_tikerr(k(1)),betaa_tikerr(l(1))];

[betaa_tikg,fktik_g,xxtik_g,time_tikg,PSNRtik_g,SSIMtik_g,betaavectik_g,ff_tikg]=tikgnSTVimplicit2d(2,500,60,x0,1,beta0_tik,bb,...
H_FT,tol,xF,maxitpcg,tolpcg,tol_r,toln,sigma1,sigma2,sigma_mat,rel_res_sf,M,k,l);

betaa_tikg=[betaa_tikg(k(1)),betaa_tikg(l(1))];

[betaa_tikW,fktik_W,xxtik_W,time_tikW,PSNRtik_W,SSIMtik_W,betaavectik_W,ff_tikW]=tikgnSTVimplicit2d(3,500,60,x0,1,beta0_tik,bb,...
H_FT,tol,xF,maxitpcg,tolpcg,tol_r,toln,sigma1,sigma2,sigma_mat,rel_res_sf,M,k,l);

betaa_tikW=[betaa_tikW(k(1)),betaa_tikW(l(1))];

save("tikpeppers.mat")
