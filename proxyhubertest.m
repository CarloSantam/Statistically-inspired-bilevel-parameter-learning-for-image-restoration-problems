clear all
close all
clc

epsi=10^(-3);


x=-100:0.001:100;

b=1;

v=1;

fun=@(x) huber_0(x,epsi)+(b/2)*(x-v).^2;

[k,l]=min(fun(x));

plot(x,fun(x),'LineWidth',1)

hold on

plot(x(l),k,'*','LineWidth',1)

%m=proxyhuber(b,epsi,v)

p=-(3/(2*epsi)+b)*2*epsi^3;

q=2*b*epsi^3*v;

dd=q^2/4+p^3/27

m=proxyhuber(b,epsi,v);

plot(m,fun(m),'o','LineWidth',1)

legend('g_\epsilon(x)','minimum point on grid','proxy')


K=-2:0.0001:2;

proxyy=@(x)proxyhuber(b,epsi,x);

for i=1:length(K)
    prox(i)=proxyy(K(i));
end

figure

plot(K,prox,'Linewidth',1)

A1 = epsi;
A2 = b;
title(sprintf('Proximal operator for Huber loss with %c=%d. and %c=%d.',949,A1,946,A2));

legend('proxy_{h_{\epsilon} \beta}(v)')


function g=proxyhuber(b,epsi,v)

p=-(3/(2*epsi)+b)*2*epsi^3;
q=2*b*epsi^3*v;
g=-2*sqrt(-p)/sqrt(3)*sin(1/3*asin((3*q*sqrt(3)/(2*p*sqrt(-p)))));

g(v>=epsi+1/b)=v-1/b;
g(v<=-epsi-1/b)=v+1/b;
end

function [L]=huber_0(y,epsi)

absy=abs(y);

c1=-1/(8*epsi^3);

c2=3/(4*epsi);

L=(y.^2).*(c1*y.^2+c2);

L(absy>epsi)=absy(absy>epsi)-(3*epsi)/8;

end