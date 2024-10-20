function [L]=huber_2(y,epsi)

absy=abs(y);

L=-(3/(2*epsi^3))*y.^2+3/(2*epsi);

L(absy>epsi)=0;

end