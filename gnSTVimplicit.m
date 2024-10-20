
function [betaa,fk,xstar,time,PNSR,SSIM]=gnSTVimplicit(type,maxit,maxit1,x_0,alpha,beta_0,bb,...
    epsi,H_FT,tol,xF,maxitminres,tolminres,sigma,tol_r,toln,rel_res_sf)
    
betaa=beta_0; %inizializziamo \beta

HTH_FT=(conj(H_FT).*H_FT); %viene dato in "pasto" a nonsmoothgradientimplicit

%minustwoHTb=-2*real(ifft2(conj(H_FT).*fft2(bb)));

%[Jac,f,ss]=nonsmoothgradientimplicit(type,epsi,H_FT,HTH_FT,betaa,...
    %x_0,xF,bb,maxitpcg,tolpcg,sigma,[]);

fk=nan(maxit1,1); %preallocamento di fk, rappresentante la funzione di ... 
%decrescita della loss

time=nan(maxit1,1); %preallocamento del tempo cumulativo

time(1)=0; %inizializziamo il tempo cumulativo

PNSR=nan(maxit1,1); %stessa cosa per PSNR e SSIM

SSIM=nan(maxit1,1);

xstar=x_0; %inizializziamo xstar, per nesterov

ss=[]; %come dicevo in nonsmoothgradient implicit, inizializziamo la variabile ss a [].

for i=1:maxit1
 
    tstart=tic;
    xstar=nesterovdescentgradient(maxit,xstar,exp(betaa),bb,epsi,H_FT,toln);
    [Jac,f,ss,relres]=nonsmoothgradientimplicit(type,epsi,H_FT,HTH_FT,betaa,xstar,xF,bb,maxitminres,tolminres,sigma,ss);
    
    %relres serve per far sì che tolminres sia variabile iterazione per
    %iterazione, dividendo ciascuna volta per rel_res.
    %rel_res_sf è una costante, che nei test
    %fissiamo a priori, per esempio in tutta la tesi ho scelto 2, partendo
    %da una tolleranza pari a 10^{-1} nel minres

    normf2=norm(f)^2;

    tolminres=relres/rel_res_sf; 
    
    d=(Jac'*f)/(-Jac'*Jac); %abbiamo sempre una divisione, visto che il
    %sistema lineare si riduce a un prodotto.
    
    betaa_new=betaa+alpha*d %svolgiamo una iterazione di GN

    a=toc(tstart);
    
    time(i+1)=time(i)+a;
    fk(i+1)=norm(f)^2;
    PNSR(i+1)=psnr(xF,xstar);
    SSIM(i+1)=ssim(xF,xstar);


    if abs(normf2-fk(i))/normf2<tol_r || norm(d)<tol 
        break
    end

    betaa=betaa_new;
    
end

time=rmmissing(time(2:end));
fk=rmmissing(fk);
PNSR=rmmissing(PNSR);
SSIM=rmmissing(SSIM);

end
