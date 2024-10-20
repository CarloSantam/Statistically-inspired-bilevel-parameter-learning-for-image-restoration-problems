function [betaa,fk,xstar,time,PNSR,SSIM,betaavec,ff]=gnSTVimplicit2d(type,maxit,maxit1,x0,alpha,beta_0,bb,...
    epsi,H_FT,tol,xF,maxitminres,tolminres,tol_r,toln,sigma1,sigma2,sigma_mat,rel_res_sf,M,k,l)

%i commenti sono totalmente analoghi al caso del parametro visto come
%scalare. Le uniche differenze che  tengo a notare sono
% - diversamente da prima calcoliamo anche betaavec, ovvero la matrice
% 2x(numero di passi svolti dall'algoritmo) dove, alla colonna i-esima, si
% calcola il parametro al passo i-esimo.
% -stavolta dobbiamo risolvere un sistema lineare. Per essere sicuri al 100
% per 100 che sia sempre invertibile si può volendo risovere
% (Jac'*Jac+10^(-6)*eye(n))\(-Jac'*f) (ad esempio), 10^(-6) è preso a caso,
% in generale comunque si può prendere un numero infinitamente piccolo.
% Alternativamente si può usare anche minres (come faccio per tikhonov con 2
% parametri). Nel codice, non avendo notato differenze di prestazione (ho
% testato tutte le varianti) semplicemente risolvo (Jac'*Jac)\(-Jac'*f).
% - betaa viene trattato non come un vettore ma come una immagine.
    betaa=beta_0(1)*M+beta_0(2)*(ones(size(M))-M);    
    
    
    fk=nan(maxit1,1);
    time=nan(maxit1,1);
    time(1)=0;
    PNSR=nan(maxit1,1);
    SSIM=nan(maxit1,1);

    xstar=x0;

    ss1=[];
    ss2=[];

    betaavec=nan(2,maxit1);
    

    for i=1:maxit1


        tstart=tic;

        [xstar,ff]=nesterovdescentgradientmp(maxit,xstar,exp(betaa),bb,epsi,H_FT,toln);
       
        [Jac,f,ss1,ss2,resf,resf2]=nonsmoothgradientimplicit2p(type,epsi,H_FT,betaa,xstar,xF,bb,maxitminres,tolminres,ss1,ss2,sigma1,sigma2,M,sigma_mat,k,l);
        betaavec(:,i+1)=[betaa(k(1));betaa(l(1))];

        normf2=norm(f)^2;
        
        d=(Jac'*Jac)\(-Jac'*f)

        dd=d(1)*M+d(2)*(ones(size(M))-M);

        betaa_new=betaa+alpha*dd;

        tolminres=min(resf/rel_res_sf,resf2/rel_res_sf);
                
                
        a=toc(tstart);
        
        time(i+1)=time(i)+a;
        PNSR(i+1)=psnr(xF,xstar);
        SSIM(i+1)=ssim(xF,xstar);
        fk(i+1)=normf2;

        if abs(normf2-fk(i))/(normf2)<tol_r || norm(d)<tol          
            break
        end

        betaa=betaa_new;
        
    end

    time=rmmissing(time(2:end));
    PNSR=rmmissing(PNSR);
    SSIM=rmmissing(SSIM);
    fk=rmmissing(fk);
    betaavec=rmmissing(betaavec,2);
    
end
