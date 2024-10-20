close all
clear all
clc

load("tikpeppers.mat")
maxitminres=100000;
tolminres=10^(-1);
tol_r=10^(-6);
toln=10^(-7);
rel_res_sf=2;
tol=10^(-3);
%k=find(M==1);
%l=find(M==0);
beta01=[0,0];
beta02=[0,1];
% x0 inizializzato con x0=bb (tutte le inizializzazioni presenti in
% "save_grigliatura.m".
[betaa_err,fk1,x1,time1,PSNR1,SSIM1,betaavec_err]=gnSTVimplicit2d(1,500,60,x0,1,beta01,bb,...
   epsi,H_FT,tol,xF,maxitminres,tolminres,tol_r,toln,sigma1,sigma2,sigma_mat,rel_res_sf,M,k,l); 
%MSE partendo da Gauss-Newton con \beta_0=[0,0]
[betaa_err1,fk12,x12,time12,PSNR12,SSIM12,betaavec_err2]=gnSTVimplicit2d(1,500,60,x0,1,betaa_tikerr,bb,...
    epsi,H_FT,tol,xF,maxitminres,tolminres,tol_r,toln,sigma1,sigma2,sigma_mat,rel_res_sf,M,k,l);
%MSE partendo da Gauss-Newton con \beta_tik
[betaa_,fk2,x2,time2,PSNR2,SSIM2,betaavec_g]=gnSTVimplicit2d(2,500,60,x0,1,beta01,bb,...
    epsi,H_FT,tol,xF,maxitminres,tolminres,tol_r,toln,sigma1,sigma2,sigma_mat,rel_res_sf,M,k,l);
%Gaussianity partendo da Gauss-Newton con \beta_0=[0,0]
[betaa_2,fk22,x22,time22,PSNR22,SSIM22,betaavec_g2]=gnSTVimplicit2d(2,500,60,x0,1,betaa_tikg,bb,...
    epsi,H_FT,tol,xF,maxitminres,tolminres,tol_r,toln,sigma1,sigma2,sigma_mat,rel_res_sf,M,k,l);
%Gaussianity partendo da Gauss-Newton con \beta_tik
[betaa,fk3,x3,time3,PSNR3,SSIM3,betaavec]=gnSTVimplicit2d(3,500,60,x0,1,beta01,bb,...
    epsi,H_FT,tol,xF,maxitminres,tolminres,tol_r,toln,sigma1,sigma2,sigma_mat,rel_res_sf,M,k,l);
%Standarised Whiteness partendo da Gauss-Newton con \beta_0=[0,0]
[betaa1,fk31,x31,time31,PSNR31,SSIM31,betaavec1]=gnSTVimplicit2d(3,500,60,x0,1,beta02,bb,...
    epsi,H_FT,tol,xF,maxitminres,tolminres,tol_r,toln,sigma1,sigma2,sigma_mat,rel_res_sf,M,k,l);
%Standarised Whiteness partendo da Gauss-Newton con \beta_0=[0,1]
[betaa2,fk32,x32,time32,PSNR32,SSIM32,betaavec2]=gnSTVimplicit2d(3,500,60,x0,1,betaa_tikW,bb,...
    epsi,H_FT,tol,xF,maxitminres,tolminres,tol_r,toln,sigma1,sigma2,sigma_mat,rel_res_sf,M,k,l);
%Standarised Whiteness partendo da Gauss-Newton con \beta_tik
[betaann,fk3nn,x3nn,time3nn,PSNR3nn,SSIM3nn,betaavecnn]=gnSTVimplicit2d(3,500,60,x0,1,beta01,bb,...
    epsi,H_FT,tol,xF,maxitminres,tolminres,tol_r,toln,sigma1,sigma2,ones(n),rel_res_sf,M,k,l);
%Non Standarised Whiteness partendo da Gauss-Newton con \beta_0=[0,0]
[betaa1nn,fk31nn,x31nn,time31nn,PSNR31nn,SSIM31nn,betaavec1nn]=gnSTVimplicit2d(3,500,60,x0,1,beta02,bb,...
    epsi,H_FT,tol,xF,maxitminres,tolminres,tol_r,toln,sigma1,sigma2,ones(n),rel_res_sf,M,k,l);
%Non Standarised Whiteness partendo da Gauss-Newton con \beta_0=[0,1]
%save("bilivelcompleman.mat")

save("bilivelcompletepeppers.mat")

%%
clear all
close all
clc

G="Test con 2 parametri\bilivelcompleteman.mat"; %"bilivel" non si può guardare... scusate
%l'errore grammaticale.
%lo chiamo G perchè a un certo punto c'è un if che fa operazioni differenti
%a seconda del nome di G (ovvero a seconda che si tratti di peppers o man)

load(G)

[minimo_err]=min(err(:));

[mu2_caerr, mu1_caerr] = find(err == minimo_err);

%troviamo il minimo su grigliatura di MSE

%linee di livello di MSE, verifichiamo che l'argmin (assoluto) coincida con
%il punto a cui converge Gauss-Newton

figure
contourf(log(MU1),log(MU2),log(err),'LineWidth',1,'EdgeColor','none')
hold on
scatter(log(mu1(mu1_caerr)),log(mu2(mu2_caerr)),'r*','LineWidth',1)
hold on
scatter(betaa_err(k(1)),betaa_err(l(1)),'o','LineWidth',1)
hold on
scatter(betaa_err1(k(1)),betaa_err1(l(1)),'o','LineWidth',1)
hold on
plot(betaavec_err(1,:),betaavec_err(2,:),'k .-','LineWidth',1)
hold on
plot(betaavec_err2(1,:),betaavec_err2(2,:),'r .-','LineWidth',1)

legend("MSE($\beta_1$,$\beta_2$)","MSE($\beta_1^*$,$\beta_2^*$) (grid-search)","MSE($\beta_1^*$,$\beta_2^*$) (Gauss-Newton)","MSE($\beta_1^*$,$\beta_2^*$) (Gauss-Newton from $\beta_{Tik}$)","Gauss-Newton trajectory","Gauss-Newton from $\beta_{Tik}$ trajectory",'Location','southwest','Interpreter','Latex')

xlabel("$\beta_1$",'Interpreter','Latex')
ylabel("$\beta_2$",'Interpreter','Latex')

title("MSE level lines")
colorbar

% plottiamo l'MSE in 3D.
figure

surfc(log(MU1),log(MU2),log(err),'LineWidth',1,'EdgeColor', 'none')
%hold on
%contour3(log(MU1),log(MU2),log(err),'LineWidth',1)
hold on
plot3(log(mu1(mu1_caerr)),log(mu2(mu2_caerr)),log(err(mu2_caerr,mu1_caerr)),'r*','LineWidth',1)
hold on
plot3(betaa_err(k(1)),betaa_err(l(1)),log(1/2*fk1(end)),'o','LineWidth',1)
hold on
plot3(betaa_err1(k(1)),betaa_err1(l(1)),log(1/2*fk12(end)),'o','LineWidth',1)
hold on
plot3((betaavec_err(1,:)),(betaavec_err(2,:)),log(1/2*fk1),'k .-','LineWidth',1)
hold on
plot3(betaavec_err2(1,:),betaavec_err2(2,:),log(1/2*fk12),'r .-','LineWidth',1)


xlabel("$\beta_1$",'Interpreter','Latex')
ylabel("$\beta_2$",'Interpreter','Latex')
zlabel("$log(MSE(\beta_1$,$\beta_2))$",'Interpreter','Latex')

title("MSE 3d plot")

legend("MSE($\beta_1$,$\beta_2$)","MSE($\beta_1^*$,$\beta_2^*$) level lines","MSE($\beta_1^*$,$\beta_2^*$) (grid-search)","MSE($\beta_1^*$,$\beta_2^*$) (Gauss-Newton)","MSE($\beta_1^*$,$\beta_2^*$) (Gauss-Newton from $\beta_{Tik}$)","Gauss-Newton trajectory","Gauss-Newton from $\beta_{Tik}$ trajectory",'Interpreter','Latex')

[minimo_g]=min(g(:));

[mu2_cag, mu1_cag] = find(g == minimo_g);

%troviamo il punto di minimo della gaussianity su grigliatura (ricordiamo
%che, per come è descritta la loss (terribilmente a cuspide), il punto di
%minimo non appartiene alla loss.

%linee di livello di MSE, verifichiamo che l'argmin (assoluto) coincida con
%il punto a cui converge Gauss-Newton.

figure
contourf(log(MU1),log(MU2),log(g),'LineWidth',1,'EdgeColor','none')
hold on
scatter(log(mu1(mu1_cag)),log(mu2(mu2_cag)),'r*','LineWidth',1)
hold on
scatter(betaa_(k(1)),betaa_(l(1)),'o','Linewidth',1)
hold on
scatter(betaa_2(k(1)),betaa_2(l(1)),'o','Linewidth',1)
hold on
plot(betaavec_g(1,:),betaavec_g(2,:),'k .-','LineWidth',1)
hold on
plot(betaavec_g2(1,:),betaavec_g2(2,:),'r .-','LineWidth',1)
colorbar

legend("g($\beta_1$,$\beta_2$)","g($\beta_1^*$,$\beta_2^*$) (grid-search)","g($\beta_1^*$,$\beta_2^*$) (Gauss-Newton)","g($\beta_1^*$,$\beta_2^*$) (Gauss-Newton from $\beta_{Tik}$)","Gauss-Newton trajectory","Gauss-Newton from $\beta_{Tik}$ trajectory",'location','southeast','Interpreter','Latex')

xlabel("$\beta_1$",'Interpreter','Latex')
ylabel("$\beta_2$",'Interpreter','Latex')

title("Gaussianity level lines")

%Gaussianity in 3D
figure
colorbar
surfc(log(MU1),log(MU2),log(g),'LineWidth',1,'EdgeColor', 'none')
%hold on
%contour3(log(MU1),log(MU2),log(g),'LineWidth',1)
hold on
plot3(log(mu1(mu1_cag)),log(mu2(mu2_cag)),log(g(mu2_cag,mu1_cag)),'r*','LineWidth',1)
hold on
plot3((betaa_(k(1))),(betaa_(l(1))),log(1/2*fk2(end)),'o','LineWidth',1)
hold on
plot3((betaa_2(k(1))),(betaa_2(l(1))),log(1/2*fk22(end)),'o','LineWidth',1)
hold on
plot3((betaavec_g(1,:)),(betaavec_g(2,:)),log(1/2*fk2),'k .- ','LineWidth',1)
hold on
plot3((betaavec_g2(1,:)),(betaavec_g2(2,:)),log(1/2*fk22),'r .- ','LineWidth',1)


xlabel("$\beta_1$",'Interpreter','Latex')
ylabel("$\beta_2$",'Interpreter','Latex')
zlabel("$log(g(\beta_1,\beta_2))$",'Interpreter','Latex')

legend("g($\beta_1$,$\beta_2$)","g($\beta_1^*$,$\beta_2^*$) level lines","g($\beta_1^*$,$\beta_2^*$) (grid-search)","g($\beta_1^*$,$\beta_2^*$) (Gauss-Newton)","g($\beta_1^*$,$\beta_2^*$) (Gauss-Newton from $\beta_{Tik}$)","Gauss-Newton trajectory","Gauss-Newton from $\beta_{Tik}$ trajectory",'Interpreter','Latex')

title("Gaussianity 3d plot")

[minimo] = min(W(:));

[mu2_ca, mu1_ca] = find(W == minimo);

[minimo2] = min(nonnormalizedW(:));

[mu2_cann, mu1_cann] = find(nonnormalizedW == minimo2);

%troviamo i punti di minimo di grigliatura della Whiteness (standardizzata
%e non). Ricordo: nonnormalizedW è sbagliato (anzi, terribilmente
%sbagliato). E' stato solamente un errore di scrittura, in realtà è
%nonstandarised. Tuttavia, è sbagliato solo il nome, l'implementazione è
%corretta. 

%a=betaa(:);
%b=betaa1(:);
k1=k(1);
l1=l(1);

%linee di livello della standarised Whiteness

figure

contourf(log(MU1),log(MU2),log(W),'LineWidth',1,'EdgeColor','none')
hold on
scatter(log(mu1(mu1_ca)),log(mu2(mu2_ca)),'r*','LineWidth',1)
hold on
scatter(betaa(k(1)),betaa(l(1)),'o','LineWidth',1)
hold on
scatter(betaa1(k(1)),betaa1(l(1)),'o','LineWidth',1)
hold on
scatter(betaa2(k(1)),betaa2(l(1)),'o','LineWidth',1)
hold on
plot(betaavec(1,:),betaavec(2,:),'k .-','LineWidth',1)
hold on
plot(betaavec1(1,:),betaavec1(2,:),'g .-','LineWidth',1)
hold on
plot(betaavec2(1,:),betaavec2(2,:),'r .-','LineWidth',1)


legend("$\widetilde{W}(\beta_1," + ...
    "\beta_2)$","$\widetilde{W}(\beta_1^*" + ...
    "\beta_2^*)$ (grid-search)","$\widetilde{W}(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0$=["+num2str(beta01(1))+","+num2str(beta01(2))+"])","$\widetilde{W}(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0=["+num2str(beta02(1))+","+num2str(beta02(2))+"]$)","$\widetilde{W}(\beta_1^*,\beta_2^*)$ (Gauss-Newton from $\beta_{Tik}$)",...
    "Gauss-Newton trajectory with $\beta_0=["+num2str(beta01(1))+","+num2str(beta01(2))+"]$", "Gauss-Newton trajectory with $\beta_0=["+num2str(beta02(1))+","+num2str(beta02(2))+"]$",'Gauss-Newton from $\beta_{Tik}$ trajectory','Interpreter', 'latex','Location','southwest')


xlabel("$\beta_1$",'Interpreter','Latex')
ylabel("$\beta_2$",'Interpreter','Latex')

title("Standarised Whiteness level lines")

%3d plot della whiteness

figure
surfc(log(MU1),log(MU2),log(W),'LineWidth',1,'EdgeColor', 'none')
%hold on
%contour3(log(MU1),log(MU2),log(W),'LineWidth',1)
hold on
plot3(log(mu1(mu1_ca)),log(mu2(mu2_ca)),log(W(mu2_ca,mu1_ca)),'r*','LineWidth',1)
hold on
plot3(betaa(k(1)),betaa(l(1)),log(fk3(end)),'o','LineWidth',1)
hold on
plot3(betaa1(k(1)),betaa1(l(1)),log(fk31(end)),'o','LineWidth',1)
hold on
plot3(betaa2(k(1)),betaa2(l(1)),log(fk32(end)),'o','LineWidth',1)
hold on
plot3(betaavec(1,1:end),betaavec(2,1:end),log(fk3),'k .-','LineWidth',1)
hold on
plot3(betaavec1(1,1:end),betaavec1(2,1:end),log(fk31),'g .-','LineWidth',1)
hold on
plot3(betaavec2(1,1:end),betaavec2(2,1:end),log(fk32),'r .-','LineWidth',1)



legend("$\widetilde{W}(\beta_1,\beta_2)$","$\widetilde{W}(\beta_1,\beta_2)$ level lines","$\widetilde{W}(\beta_1^*," + ...
    "\beta_2^*)$ (grid-search)","$\widetilde{W}(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0=["+num2str(beta01(1))+","+num2str(beta01(2))+"]$)","$\widetilde{W}(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0=["+num2str(beta02(1))+","+num2str(beta02(2))+"]$)","$\widetilde{W}(\beta_1^*,\beta_2^*)$ (Gauss-Newton from $\beta_{Tik}$)",...
    "Gauss-Newton trajectory with $\beta_0=["+num2str(beta01(1))+","+num2str(beta01(2))+"]$", "Gauss-Newton trajectory with $\beta_0=["+num2str(beta02(1))+","+num2str(beta02(2))+"]$","Gauss-Newton from $\beta_{Tik}$ trajectory",'Interpreter', 'latex')

xlabel("$\beta_1$",'Interpreter','Latex')
ylabel("$\beta_2$",'Interpreter','Latex')
zlabel("$log(\widetilde{W}(\beta_1,\beta_2))$",'Interpreter', 'latex')

title("Standarised Whiteness 3d plot")

%linee di livello della non standarised Whiteness

figure

contourf(log(MU1),log(MU2),log(nonnormalizedW),'LineWidth',1,'EdgeColor','none')
hold on
scatter(log(mu1(mu1_cann)),log(mu2(mu2_cann)),'r*','LineWidth',1)
hold on
scatter(betaann(k(1)),betaann(l(1)),'o','LineWidth',1)
hold on
scatter(betaa1nn(k(1)),betaa1nn(l(1)),'o','LineWidth',1)
plot(betaavecnn(1,:),betaavecnn(2,:),'k .-','LineWidth',1)
hold on
plot(betaavec1nn(1,:),betaavec1nn(2,:),'g .-','LineWidth',1)


legend("$W(\beta_1,\beta_2)$","$W(\beta_1^*,\beta_2^*)$ (grid-search)","$W(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0=["+num2str(beta01(1))+","+num2str(beta01(2))+"]$)","$W(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0=["+num2str(beta02(1))+","+num2str(beta02(2))+"]$)","Gauss-Newton trajectory with $\beta_0=["+num2str(beta01(1))+","+num2str(beta01(2))+"]$", "Gauss-Newton trajectory with $\beta_0=["+num2str(beta02(1))+","+num2str(beta02(2))+"]$",'Interpreter','Latex',...
    'Location','southwest')

xlabel("$\beta_1$",'Interpreter','Latex')
ylabel("$\beta_2$",'Interpreter','Latex')

title("Non Standarised Whiteness level lines")

% 3d plot della non standarised whiteness
figure
surfc(log(MU1),log(MU2),log(nonnormalizedW),'LineWidth',1,'EdgeColor', 'none')
%hold on
%contour3(log(MU1),log(MU2),log(nonnormalizedW),'LineWidth',1)
hold on
plot3(log(mu1(mu1_cann)),log(mu2(mu2_cann)),log(nonnormalizedW(mu2_cann,mu1_cann)),'r*','LineWidth',1)
hold on
plot3(betaann(k(1)),betaann(l(1)),log(fk3nn(end)),'o','LineWidth',1)
hold on
plot3(betaa1nn(k(1)),betaa1nn(l(1)),log(fk31nn(end)),'o','LineWidth',1)

hold on
plot3(betaavecnn(1,1:end),betaavecnn(2,1:end),log(fk3nn(1:end)),'k .-','LineWidth',1)
hold on
plot3(betaavec1nn(1,1:end),betaavec1nn(2,1:end),log(fk31nn(1:end)),'g .-','LineWidth',1)


legend("$W(\beta_1,\beta_2)$","$W(\beta_1,\beta_2)$ level lines","$W(\beta_1^*,\beta_2^*)$ (grid-search)","$W(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0=["+num2str(beta01(1))+","+num2str(beta01(2))+"]$)","$W(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0=["+num2str(beta02(1))+","+num2str(beta02(2))+"]$)","Gauss-Newton trajectory with $\beta_0=["+num2str(beta01(1))+","+num2str(beta01(2))+"]$", "Gauss-Newton trajectory with $\beta_0=["+num2str(beta02(1))+","+num2str(beta02(2))+"]$",'Interpreter','Latex')

xlabel("$\beta_1$",'Interpreter','Latex')
ylabel("$\beta_2$",'Interpreter','Latex')
zlabel("$log(W(\beta_1,\beta_2))$",'Interpreter','Latex')


title("Non Standarised Whiteness 3d plot")


figure

maxpsnr=max(PSNR(:));

[m2_p,m1_p]=find(PSNR==maxpsnr);

%troviamo il massimo del PSNR e plottiamo le linee di livello del PSNR, in
%modo tale da verificare che il PSNR ottenuto attraverso la risolvuzione
%dell'ottimizzazione bilivello delle 3 loss si avvicini il più possibile al
%massimo. Nel caso dell'SSIM, si farà lo stesso.
%N.B.: qui ci sono solamente i punti di minimo GLOBALI, per un paragone
%tra i punti di minimo solo locali, c'è un plot dopo.

contourf(log(MU1),log(MU2),(PSNR),'LineWidth',1,'EdgeColor','none')

hold on

scatter(log(mu1(m1_p)),log(mu2(m2_p)),'r*','LineWidth',1)

hold on

scatter(betaa_err(k1),betaa_err(l1),'o','LineWidth',1)

hold on

scatter(betaa_(k1),betaa_(l1),'o','LineWidth',1)

hold on

scatter(betaa1(k1),betaa1(l1),'o','LineWidth',1)

hold on

scatter(betaa1nn(k1),betaa1nn(l1),'o','LineWidth',1)

legend("PSNR($\beta_1,\beta_2$)","max(PSNR)","PSNR (MSE)","PSNR (Gaussianity)","PSNR ($\widetilde{W}$)","PSNR (W)",'Location','southwest','Interpreter','Latex')

title("PSNR level lines")

figure

surfc(log(MU1),log(MU2),PSNR,'Linewidth',1,'Edgecolor','none')

%hold on

%contour3(log(MU1),log(MU2),PSNR,'Linewidth',1)

hold on

plot3(log(mu1(m1_p)),log(mu2(m2_p)),maxpsnr,'r*','Linewidth',1)

hold on

plot3((betaa_err(k(1))),(betaa_err(l(1))),PSNR1(end),'o','Linewidth',1)

hold on

plot3((betaa_(k(1))),(betaa_(l(1))),PSNR2(end),'o','Linewidth',1)

hold on

plot3((betaa1(k(1))),(betaa1(l(1))),PSNR31(end),'o','Linewidth',1)

hold on

plot3((betaa1nn(k(1))),(betaa1nn(l(1))),PSNR31nn(end),'o','Linewidth',1)

legend("PSNR($\beta_1,\beta_2$)","PSNR level lines","max(PSNR)","PSNR (MSE)","PSNR (Gaussianity)","PSNR ($\widetilde{W}$)","PSNR (W)",'Location','southwest','Interpreter','Latex')

title("PSNR 3d plot")

xlabel("$\beta_1$",'Interpreter','latex')
ylabel("$\beta_2$",'Interpreter','latex')
zlabel('$PSNR(\beta_1,\beta_2)$','interpreter','latex')

% QUI VERIFICHIAMO IL PSNR (E POI SSIM) DEI RISULTATI OTTENUTI TRAMITE
% OTTIMIZZAZIONE BILIVELLO PER WHITENES E NON STANDARISED WHITENESS.VA
% RICORDATO CHE ALCUNI PARAMETRI OTTENUTI TRAMITE BILIVELLO SONO SOLO P.TI DI 
% MINIMO LOCALI E NON GLOBALI. QUI MOSTRIAMO CHE IL PSNR (E POI SSIM) SI
% ALLONTANO TANTO DAL P.TO DI MINIMO MASSIMO. (l'ho scritto in maiuscolo,
% perchè ritengo molto importante questi plot).

figure

contourf(log(MU1),log(MU2),(PSNR),'LineWidth',1,'EdgeColor','none')

hold on

scatter(betaa(k1),betaa(l1),'o','LineWidth',1)

hold on

scatter(betaa1(k1),betaa1(l1),'*','LineWidth',1)

hold on

scatter(betaann(k1),betaann(l1),'k o','LineWidth',1)

hold on

scatter(betaa1nn(k1),betaa1nn(l1),'*','LineWidth',1)

hold on

scatter(betaa2(k(1)),betaa2(l(1)),'+','LineWidth',1)

hold on

plot(betaavec(1,:),betaavec(2,:),'r .-','LineWidth',1)
hold on
plot(betaavec1(1,:),betaavec1(2,:),'r :','LineWidth',1)
hold on
plot(betaavecnn(1,:),betaavecnn(2,:),'k .-','LineWidth',1)
hold on
plot(betaavec1nn(1,:),betaavec1nn(2,:),'k :','LineWidth',1)
hold on
plot(betaavec2(1,:),betaavec2(2,:),'b .-','LineWidth',1)

title("PSNR level lines")

xlabel("$\beta_1$",'Interpreter','Latex')
ylabel("$\beta_2$",'Interpreter','Latex')

legend('PSNR($\beta_1$,$\beta_2)$',"$\widetilde{W}(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0=["+num2str(beta01(1))+","+num2str(beta01(2))+"]$)","$\widetilde{W}(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0=["+num2str(beta02(1))+","+num2str(beta02(2))+"]$)",...
    "$W(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0=["+num2str(beta01(1))+","+num2str(beta01(2))+"]$)","$W(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0=["+num2str(beta02(1))+","+num2str(beta02(2))+"]$)","$\widetilde{W}(\beta_1^*,\beta_2^*)$ (Gauss-Newton from $\beta_{Tik}$)","Gauss-Newton trajectory for $\widetilde{W}$ with $\beta_0=["+num2str(beta01(1))+","+num2str(beta01(2))+"]$", "Gauss-Newton trajectory for $\widetilde{W}$ with $\beta_0=["+num2str(beta02(1))+","+num2str(beta02(2))+"]$",...
    "Gauss-Newton trajectory for $W$ with $\beta_0=["+num2str(beta01(1))+","+num2str(beta01(2))+"]$","Gauss-Newton trajectory for $W$ with $\beta_0=["+num2str(beta02(1))+","+num2str(beta02(2))+"]$","Gauss-Newton for $\widetilde{W}$ from $\beta_{Tik}$ trajectory",'Interpreter','Latex','Location','southeast')


figure

surfc(log(MU1),log(MU2),(PSNR),'LineWidth',1,'EdgeColor', 'none')
%hold on
%contour3(log(MU1),log(MU2),(PSNR),'LineWidth',1)
hold on
plot3(betaa(k(1)),betaa(l(1)),(PSNR3(end)),'o','LineWidth',1)
hold on
plot3(betaa1(k(1)),betaa1(l(1)),(PSNR31(end)),'*','LineWidth',1)
hold on
plot3(betaann(k(1)),betaann(l(1)),(PSNR3nn(end)),'k o','LineWidth',1)
hold on
plot3(betaa1nn(k(1)),betaa1nn(l(1)),(PSNR31nn(end)),'*','LineWidth',1)


legend('PSNR($\beta_1$,$\beta_2$)','PSNR level lines',"$\widetilde{W}(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0=["+num2str(beta01(1))+","+num2str(beta01(2))+"]$)","$\widetilde{W}(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0=["+num2str(beta02(1))+","+num2str(beta02(2))+"]$)",...
    "$W(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0=["+num2str(beta01(1))+","+num2str(beta01(2))+"]$)","$W(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0=["+num2str(beta02(1))+","+num2str(beta02(2))+"]$)",'Interpreter','Latex','Location','southwest')

title("PSNR")


xlabel("$\beta_1$",'Interpreter','Latex')
ylabel("$\beta_2$",'Interpreter','Latex')
zlabel("$PSNR(\beta_1,\beta_2)$",'Interpreter','Latex')

title("PSNR 3d plot")


figure

contourf(log(MU1),log(MU2),(SSIM),'LineWidth',1,'EdgeColor','none')

hold on

scatter(betaa(k1),betaa(l1),'o','LineWidth',1)

hold on

scatter(betaa1(k1),betaa1(l1),'*','LineWidth',1)

hold on

scatter(betaann(k1),betaann(l1),'k o','LineWidth',1)

hold on

scatter(betaa1nn(k1),betaa1nn(l1),'*','LineWidth',1)

hold on

scatter(betaa2(k(1)),betaa2(l(1)),'o','LineWidth',1)

hold on

plot(betaavec(1,:),betaavec(2,:),'r .-','LineWidth',1)
hold on
plot(betaavec1(1,:),betaavec1(2,:),'r :','LineWidth',1)
hold on
plot(betaavecnn(1,:),betaavecnn(2,:),'k .-','LineWidth',1)
hold on
plot(betaavec1nn(1,:),betaavec1nn(2,:),'k :','LineWidth',1)
hold on
plot(betaavec2(1,:),betaavec2(2,:),'b .-','LineWidth',1)




legend('SSIM($\beta_1$,$\beta_2)$',"$\widetilde{W}(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0=["+num2str(beta01(1))+","+num2str(beta01(2))+"]$)","$\widetilde{W}(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0=["+num2str(beta02(1))+","+num2str(beta02(2))+"]$)",...
    "$W(\beta_1*,\beta_2*)$ (Gauss-Newton with $\beta_0=["+num2str(beta01(1))+","+num2str(beta01(2))+"]$)","$W(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0=["+num2str(beta02(1))+","+num2str(beta02(2))+"]$)","$\widetilde{W}(\beta_1^*,\beta_2^*)$ (Gauss-Newton from $\beta_{Tik}$)",...
    "Gauss-Newton trajectory for $\widetilde{W}$ with $\beta_0=["+num2str(beta01(1))+","+num2str(beta01(2))+"]$", "Gauss-Newton trajectory for $\widetilde{W}$ with $\beta_0=["+num2str(beta02(1))+","+num2str(beta02(2))+"]$",...
    "Gauss-Newton trajectory for $W$ with $\beta_0=["+num2str(beta01(1))+","+num2str(beta01(2))+"]$","Gauss-Newton trajectory for $W$ with $\beta_0=["+num2str(beta02(1))+","+num2str(beta02(2))+"]$","Gauss-Newton from $\beta_{Tik}$ trjectory for $\widetilde{W}$",'Interpreter','Latex','Location','southeast')

xlabel("$\beta_1$",'Interpreter','Latex')
ylabel("$\beta_2$",'Interpreter','Latex')

title("SSIM level lines")


figure

surfc(log(MU1),log(MU2),(SSIM),'LineWidth',1,'EdgeColor', 'none')
%hold on
%contour3(log(MU1),log(MU2),(SSIM),'LineWidth',1)
hold on
plot3(betaa(k(1)),betaa(l(1)),(SSIM3(end)),'o','LineWidth',1)
hold on
plot3(betaa1(k(1)),betaa1(l(1)),(SSIM31(end)),'*','LineWidth',1)
hold on
plot3(betaann(k(1)),betaann(l(1)),(SSIM3nn(end)),'k o','LineWidth',1)
hold on
plot3(betaa1nn(k(1)),betaa1nn(l(1)),(SSIM31nn(end)),'*','LineWidth',1)

legend('SSIM($\beta_1$,$\beta_2$)','SSIM level lines',"$\widetilde{W}(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0=["+num2str(beta01(1))+","+num2str(beta01(2))+"]$)","$\widetilde{W}(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0=["+num2str(beta02(1))+","+num2str(beta02(2))+"]$)",...
    "$W(\beta_1*,\beta_2*)$ (Gauss-Newton with $\beta_0=["+num2str(beta01(1))+","+num2str(beta01(2))+"]$)","$W(\beta_1^*,\beta_2^*)$ (Gauss-Newton with $\beta_0=["+num2str(beta02(1))+","+num2str(beta02(2))+"]$)",'Interpreter','Latex','Location','southwest')

xlabel("\beta_1")
ylabel("\beta_2")
zlabel("$SSIM(\beta_1,\beta_2)$",'Interpreter','Latex')

title("SSIM 3d plot")

% esattamente per il PSNR verichiamo che l'SSIM delle ricostruzioni per i
% le tre loss (per i punti di minimo globali) si avvicinino al punto di
% massimo dell'SSIM.
figure


maxssim=max(SSIM(:));

[m2_p,m1_p]=find(SSIM==maxssim);

contourf(log(MU1),log(MU2),(SSIM),'LineWidth',1,'EdgeColor','none')

hold on

scatter(log(mu1(m1_p)),log(mu2(m2_p)),'r*','LineWidth',1)

hold on

scatter(betaa_err(k1),betaa_err(l1),'o','LineWidth',1)

hold on

scatter(betaa_(k1),betaa_(l1),'o','LineWidth',1)

hold on

scatter(betaa1(k1),betaa1(l1),'o','LineWidth',1)

hold on

scatter(betaa1nn(k1),betaa1nn(l1),'o','LineWidth',1)

legend("SSIM($\beta_1,\beta_2$)","max(SSIM)","SSIM (MSE)","SSIM (Gaussianity)","SSIM $(\widetilde{W})$","SSIM(W)",'Location','southwest','Interpreter','Latex')

title("SSIM level lines")

figure

%3d plot dell'SSIM.

surfc(log(MU1),log(MU2),SSIM,'Linewidth',1,'Edgecolor','none')

%hold on

%contour3(log(MU1),log(MU2),SSIM,'Linewidth',1)

hold on

plot3(log(mu1(m1_p)),log(mu2(m2_p)),maxssim,'r*','Linewidth',1)

hold on

plot3((betaa_err(k(1))),(betaa_err(l(1))),SSIM1(end),'o','Linewidth',1)

hold on

plot3((betaa_(k(1))),(betaa_(l(1))),SSIM2(end),'o','Linewidth',1)

hold on

plot3((betaa1(k(1))),(betaa1(l(1))),SSIM31(end),'o','Linewidth',1)

hold on

plot3((betaa1nn(k(1))),(betaa1nn(l(1))),SSIM31nn(end),'o','Linewidth',1)

legend("SSIM($\beta_1,\beta_2$)","SSIM level lines","max(SSIM)","SSIM (MSE)","SSIM (Gaussianity)","SSIM ($\widetilde{W}$)","SSIM (W)",'Location','southwest','Interpreter','Latex')

title("SSIM 3d plot")

xlabel('$\beta_1$','interpreter','latex')
ylabel('$\beta_2$','interpreter','latex')
zlabel('$SSIM(\beta_1,\beta_2)$','interpreter','latex')

figure

%Mostriamo:
%1) l'immagine originale
%2) l'immagine corrotta
%3) la maschera dell'immagine ottenuta

subplot(1,3,1)
imshow2(xF)

title("Original image")

subplot(1,3,2)

imshow2(bb)
title("Corrupted image")

subplot(1,3,3)

imshow2(M)

title("Image mask")


if  strcmp(G,'Test con 2 parametri\bilivelcompleteman.mat')
    figure
    subplot(1,2,1)
    imshow2(x3)
    title("Local minimum","\beta=["+num2str(betaa(k(1)))+","+num2str(betaa(l(1)))+"]",'FontWeight','bold')

    subplot(1,2,2)
    imshow2(x31)
    title("Global minimum","\beta=["+num2str(betaa1(k(1)))+","+num2str(betaa1(l(1)))+"]",'FontWeight','bold')
    sgtitle("Optimal restoration by bilevel Standarised Whiteness")

    figure

    subplot(1,2,1)
    imshow2(x3nn)
    title("Local minimum","\beta=["+num2str(betaann(k(1)))+","+num2str(betaann(l(1)))+"]",'FontWeight','bold')

    subplot(1,2,2)
    imshow2(x31nn)
    title("Global minimum","\beta=["+num2str(betaa1nn(k(1)))+","+num2str(betaa1nn(l(1)))+"]",'FontWeight','bold')
    sgtitle("Optimal restoration by bilevel Non Standarised Whiteness")

elseif strcmp(G,'Test con 2 parametri\bilivelcomplepeppers.mat')

    subplot(1,2,1)
    imshow2(x31)
    title("Optimal restoration by bilevel Standarised Whiteness","\beta=["+num2str(betaa(k(1)))+","+num2str(betaa(l(1)))+"]",'FontWeight','bold')

    subplot(1,2,2)
    imshow2(x3nn)
    title("Optimal restoration by bilevel Non Standarised Whiteness","\beta=["+num2str(betaann(k(1)))+","+num2str(betaann(l(1)))+"]",'FontWeight','bold')

end

figure

subplot(1,2,1)

imshow2(x1)

title("Optimal restoration by bilevel-MSE","\beta=["+num2str(betaa_err(k(1)))+","+num2str(betaa_err(l(1)))+"]",'FontWeight','bold')

subplot(1,2,2)

imshow2(x2)

title("Optimal restoration by bilevel-Gaussianity","\beta=["+num2str(betaa_(k(1)))+","+num2str(betaa_(l(1)))+"]",'FontWeight','bold')
