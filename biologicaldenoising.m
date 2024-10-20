clear all
close all
clc

load("denoisingbiologicalimage")

[PSFtilde,~]=psfGauss([n-1,n-1],0.32);

H_FT=psf2otf(PSFtilde,[n,n]);

tolpcg=10^(-1);

tol=10^(-3);

tolr=10^(-6);

[betaa_gauss,fkg,xg,timeg,PSNRg,SSIMg,betaavec_g]=gnSTVimplicit2d(2,500,30,x0,1,[0,0],bb,epsi,H_FT,tol,bb,maxitpcg,tolpcg,tol_r,toln,sigma1,sigma2,sigma_mat,rel_res_sf,M,k,l);

[betaa_w,fkw,xw,timew,PSNRw,SSIMw,betaavec_w]=gnSTVimplicit2d(3,500,30,x0,1,[0,0],bb,epsi,H_FT,tol,bb,maxitpcg,tolpcg,tol_r,toln,sigma1,sigma2,sigma_mat,rel_res_sf,M,k,l);

[betaa_wnn,fkwnn,xwnn,timewnn,PSNRwnn,SSIMwnn,betaavec_wnn]=gnSTVimplicit2d(3,500,30,x0,1,[0,0],bb,epsi,H_FT,tol,bb,maxitpcg,tolpcg,tol_r,toln,sigma1,sigma2,ones(n),rel_res_sf,M,k,l);


%save("denoisingbiologicalimage")

%%
clear all
close all
clc

load("denoisingbiologicalimage.mat")


figure

subplot(1,2,1)

imshow2(bb)

title("Original Image")

subplot(1,2,2)

imshow2(M)

title("Mask of the image")

figure

subplot(1,3,1)

imshow2(xg)

title("Optimal restoration", "by bilevel Gaussianity",'FontWeight','bold')

subplot(1,3,2)

imshow2(xw)

title("Optimal restoration", "by bilevel Standarised Whiteness",'FontWeight','bold')

subplot(1,3,3)

imshow2(xwnn)

title("Optimal restoration", "by bilevel Non Standarised Whiteness",'FontWeight','bold')


figure

subplot(3,1,1)

plot((1/2)*fkg,'LineWidth',1)

xlabel("i")

ylabel("g_i")

title("Gaussianity decays")

subplot(3,1,2)

plot(fkw,'LineWidth',1)

ylabel("$\widetilde{W}_i$",'Interpreter','latex')

xlabel("i",'Interpreter','latex')

title("Standardised Whiteness decays")

subplot(3,1,3)

plot(fkwnn,'LineWidth',1)

ylabel("${W}_i$",'Interpreter','latex')

xlabel("i",'Interpreter','latex')

title("Non Standardised Whiteness decays")


figure

subplot(3,1,1)

plot(timeg,'LineWidth',1)

xlabel("i",'Interpreter','latex')

ylabel("s",'Interpreter','latex')

title("Gaussianity decays cumulative time")

subplot(3,1,2)

plot(timew,'LineWidth',1)

ylabel("s",'Interpreter','latex')

xlabel("i",'Interpreter','latex')

title("Standardised Whiteness cumulative time")

subplot(3,1,3)

plot(timewnn,'LineWidth',1)

ylabel("s",'Interpreter','latex')

xlabel("i",'Interpreter','latex')

title("Non Standardised Whiteness decays cumulative time")


figure

subplot(3,1,1)

plot(timeg,fkg,'LineWidth',1)

xlabel("i",'Interpreter','latex')

ylabel("s",'Interpreter','latex')

title("Gaussianity decays in cumulative time")

subplot(3,1,2)

plot(timew,fkw,'LineWidth',1)

ylabel("s",'Interpreter','latex')

xlabel("i",'Interpreter','latex')

title("Standardised Whiteness decays in cumulative time")

subplot(3,1,3)

plot(timewnn,fkwnn,'LineWidth',1)

ylabel("s",'Interpreter','latex')

xlabel("i",'Interpreter','latex')

title("Non Standardised Whiteness decays in cumulative time")
