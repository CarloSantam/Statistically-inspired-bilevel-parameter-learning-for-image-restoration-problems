function W = GRWP(H_FT,bb,HTbb_FT,HTH_FT,DTD_FT,mu)      
    sol_FT = HTbb_FT./(HTH_FT + DTD_FT/mu);
    xFF = real(ifft2(sol_FT));
    HxFF=real(ifft2(H_FT.*fft2(xFF)));
    ResF=HxFF-bb;
    Resnorm=norm(ResF(:));
    ccorrelation2=real(ifft2(fft2(ResF).*conj(fft2(ResF))));
    f=ccorrelation2(:)/Resnorm^2;
    W=norm(f)^2;
end
