function [L]=huber_0(y,epsi)

absy=abs(y);

c1=-1/(8*epsi^3);

c2=3/(4*epsi);

L=(y.^2).*(c1*y.^2+c2);

L(absy>epsi)=absy(absy>epsi)-(3*epsi)/8;

end