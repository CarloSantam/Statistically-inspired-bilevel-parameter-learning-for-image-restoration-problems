function g=gaussianity(H_FT,bb,HTbb_FT,HTH_FT,DTD_FT,mu,sigma,n)
    sol_FT = HTbb_FT./(HTH_FT + DTD_FT/mu);
    xFF = real(ifft2(sol_FT));
    HxFF=real(ifft2(H_FT.*fft2(xFF)));
    ResF=HxFF-bb;
    g=((norm(ResF(:),2).^2-sigma^2*(n^2))^2)/2;
end
