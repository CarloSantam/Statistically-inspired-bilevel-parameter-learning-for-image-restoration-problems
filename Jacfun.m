function [Jac,f,Sgrad,p,s]=Jacfun(DTD_FT,H_FT,HTH_FT,beta,b_folder1,b_folder2...
    ,b_folder3,xF_folder1,xF_folder2,xF_folder3,type,sigma1,sigma2,sigma3)

if ~ismember(type, [1, 2, 3])
    disp('This type is not avaiable')
    return
end

xF1 = dir(fullfile(xF_folder1,'*.jpg'));
xF2 = dir(fullfile(xF_folder2,'*.jpg'));
xF3 = dir(fullfile(xF_folder3,'*.jpg'));
    
b1 = dir(fullfile(b_folder1,'*.jpg'));
b2 = dir(fullfile(b_folder2,'*.jpg'));
b3= dir(fullfile(b_folder3,'*.jpg'));
    
J1=zeros(length(xF1),1);
J2=zeros(length(xF2),1);
J3=zeros(length(xF3),1);
    
f1=zeros(length(xF1),1);
f2=zeros(length(xF2),1);
f3=zeros(length(xF3),1);
a1=0;
a2=0;
a3=0;

p1=zeros(length(xF1),1);
p2=zeros(length(xF2),1);
p3=zeros(length(xF3),1);

s1=zeros(length(xF1),1);
s2=zeros(length(xF2),1);
s3=zeros(length(xF3),1);

if type==1
    
    for l=1:numel(xF1)
        
        xF = imread(fullfile(xF_folder1,xF1(l).name));
        xF=im2double(im2gray(xF));
        bb=imread(fullfile(b_folder1,b1(l).name));
        bb=im2double(im2gray(bb));
        bbhat=fft2(bb);
        [grad,f,p,s]=gradfun(DTD_FT,H_FT,HTH_FT,beta,bbhat,xF);
        J1(l)=(grad'*f)/norm(f);
        f1(l)=norm(f);
        a1=(grad)'*f+a1;
        p1(l)=p;
        s1(l)=s;
    end
    
    for l=1:numel(xF2)
        xF = imread(fullfile(xF_folder2,xF2(l).name));
        xF=im2double(im2gray(xF));
        bb=imread(fullfile(b_folder2,b2(l).name));
        bb=im2double(im2gray(bb));
        bbhat=fft2(bb);
        [grad,f,p,s]=gradfun(DTD_FT,H_FT,HTH_FT,beta,bbhat,xF);
        J2(l)=(grad'*f)/norm(f);
        f2(l)=norm(f);
        a2=(grad)'*f+a2;
        p2(l)=p;
        s2(l)=s;
    end
    
    for l=1:numel(xF3)
        xF = imread(fullfile(xF_folder3,xF3(l).name));
        xF=im2double(im2gray(xF));
        bb=imread(fullfile(b_folder3,b3(l).name));
        bb=im2double(im2gray(bb));
        bbhat=fft2(bb);
        [grad,f,p,s]=gradfun(DTD_FT,H_FT,HTH_FT,beta,bbhat,xF);
        J3(l)=(grad'*f)/norm(f);
        f3(l)=norm(f);
        a3=(grad)'*f+a3;
        p3(l)=p;
        s3(l)=s;
    end
    
    Jac=[J1;J2;J3];
    f=[f1;f2;f3];
    Sgrad=2*(a1+a2+a3);
    p=[p1;p2;p3];
    s=[s1;s2;s3];

elseif type==2
    
    for l=1:numel(xF1)
        
        xF = imread(fullfile(xF_folder1,xF1(l).name));
        xF=im2double(im2gray(xF));
        [n,~]=size(xF);
        bb=imread(fullfile(b_folder1,b1(l).name));
        bb=im2double(im2gray(bb));
        bbhat=fft2(bb);
        [grad,f]=gradfungauss(DTD_FT,H_FT,HTH_FT,beta,bbhat,n,sigma1);
        J1(l)=grad;
        f1(l)=f;
        a1=(grad)'*f+a1;
    
    end
    
    for l=1:numel(xF2)
        xF = imread(fullfile(xF_folder2,xF2(l).name));
        xF=im2double(im2gray(xF));
        [n,~]=size(xF);
        bb=imread(fullfile(b_folder2,b2(l).name));
        bb=im2double(im2gray(bb));
        bbhat=fft2(bb);
        [grad,f]=gradfungauss(DTD_FT,H_FT,HTH_FT,beta,bbhat,n,sigma2);
        J2(l)=grad;
        f2(l)=f;
        a2=(grad)'*f+a2;
    end
    
    for l=1:numel(xF3)
        xF = imread(fullfile(xF_folder3,xF3(l).name));
        xF=im2double(im2gray(xF));
        [n,~]=size(xF);
        bb=imread(fullfile(b_folder3,b3(l).name));
        bb=im2double(im2gray(bb));
        bbhat=fft2(bb);
        [grad,f]=gradfungauss(DTD_FT,H_FT,HTH_FT,beta,bbhat,n,sigma3);
        J3(l)=grad;
        f3(l)=f;
        a3=(grad)'*f+a3;
    end
    
    Jac=[J1;J2;J3];
    f=[f1;f2;f3];
    Sgrad=2*(a1+a2+a3);
    


elseif type==3
    
    for l=1:numel(xF1)
        
        xF = imread(fullfile(xF_folder1,xF1(l).name));
        xF=im2double(im2gray(xF));
        bb=imread(fullfile(b_folder1,b1(l).name));
        bb=im2double(im2gray(bb));
        bbhat=fft2(bb);
        [grad,f]=gradfunwhiteness(DTD_FT,H_FT,HTH_FT,beta,bbhat,bb);
        J1(l)=(grad'*f)/norm(f);
        f1(l)=norm(f);
        a1=(grad)'*f+a1;
    
    end
    
    for l=1:numel(xF2)
        xF = imread(fullfile(xF_folder2,xF2(l).name));
        xF=im2double(im2gray(xF));
        bb=imread(fullfile(b_folder2,b2(l).name));
        bb=im2double(im2gray(bb));
        bbhat=fft2(bb);
        [grad,f]=gradfunwhiteness(DTD_FT,H_FT,HTH_FT,beta,bbhat,bb);
        J2(l)=(grad'*f)/norm(f(:));
        f2(l)=norm(f);
        a2=(grad)'*f+a2;
    end
    
    for l=1:numel(xF3)
        xF = imread(fullfile(xF_folder3,xF3(l).name));
        xF=im2double(im2gray(xF));
        bb=imread(fullfile(b_folder3,b3(l).name));
        bb=im2double(im2gray(bb));
        bbhat=fft2(bb);
        [grad,f]=gradfunwhiteness(DTD_FT,H_FT,HTH_FT,beta,bbhat,bb);
        J3(l)=(grad'*f)/norm(f(:));
        f3(l)=norm(f);
        a3=(grad)'*f+a3;
    end
    
    Jac=[J1;J2;J3];
    f=[f1;f2;f3];
    Sgrad=2*(a1+a2+a3);
    
end

end
