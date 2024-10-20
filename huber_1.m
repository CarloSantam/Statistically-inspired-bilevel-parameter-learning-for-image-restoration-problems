function [L]=huber_1(y,epsi)

absy=abs(y);


c1=-1/(2*epsi^3);

c2=3/(2*epsi);

L=y.*(c1*y.^2+c2);

L(absy>epsi)=sign(y(absy>epsi));

end