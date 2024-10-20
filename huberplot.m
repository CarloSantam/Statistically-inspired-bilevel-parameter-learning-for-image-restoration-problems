clear all
close all
clc

epsi=1;

y=linspace(-2,2,1001);

for i=1:length(y)
    if abs(y(i))<=epsi
        L(i)=y(i)^2/(2*epsi);
    else
        L(i)=abs(y(i))-epsi/2;
end
end

plot(y,L,'Linewidth',1)
hold on

for i=1:length(y)
    if abs(y(i))<=epsi
        L(i)=-1/(8*epsi^3)*y(i)^4+3/(4*epsi)*y(i)^2;
    else
        L(i)=abs(y(i))-(3*epsi)/8;
end
end

plot(y,L,'Linewidth',1)

hold on

plot(y,abs(y),'Linewidth',1)

legend('$h_{\epsilon} \in \mathcal{C}^1$','$h_\epsilon \in \mathcal{C}^2$','$|x|$','interpreter','LaTex')

title('Two different Huber loss with \epsilon=1')