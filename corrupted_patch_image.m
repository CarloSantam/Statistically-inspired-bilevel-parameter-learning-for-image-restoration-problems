clear all 
close all
clc

randn('seed',17)
sigma1=0.001;
sigma2=0.01;
sigma3=0.1;

% Loop attraverso ogni immagine nella cartella
image_folder = 'Validation+learning\patch_validation\Sigma1';
patch_folder = 'Validation+learning\corrupted_patch_validation\Sigma1';
image_files = dir(fullfile(image_folder, '*.jpg'));
patch_size = [100,100];


for i = 1:numel(image_files)
    % Leggi l'immagine
    image = imread(fullfile(image_folder, image_files(i).name));
    image=im2double(im2gray(image));
    n=min(size(image));
    m=11;% support PSF           
    [PSFtilde,~]=psfGauss([m,m],4);
    H_FT=psf2otf(PSFtilde,[n,n]);
    b= real(ifft2(H_FT.*fft2(image)));
    noise = sigma1*randn(n,n);
    bb=b+noise;
    [~, name, ext] = fileparts(image_files(i).name);
    patch_name = sprintf('%s_.jpg', name);
    imwrite(bb,fullfile(patch_folder, patch_name),'jpg');
end


image_folder = 'Validation+learning\patch_validation\Sigma2';
patch_folder = 'Validation+learning\corrupted_patch_validation\Sigma2';
image_files = dir(fullfile(image_folder, '*.jpg'));
patch_size = [100,100];


for i = 1:numel(image_files)
    image = imread(fullfile(image_folder, image_files(i).name));
    image=im2double(im2gray(image));
    n=min(size(image));
    m=11;% support PSF           
    [PSFtilde,~]=psfGauss([m,m],4);
    H_FT=psf2otf(PSFtilde,[n,n]);
    b= real(ifft2(H_FT.*fft2(image)));
    noise = sigma2*randn(n,n);
    bb=b+noise;
    [~, name, ext] = fileparts(image_files(i).name);
    patch_name = sprintf('%s_.jpg', name);
    imwrite(bb,fullfile(patch_folder, patch_name),'jpg');
end


image_folder = 'Validation+learning\patch_validation\Sigma3';
patch_folder = 'Validation+learning\corrupted_patch_validation\Sigma3';
image_files = dir(fullfile(image_folder, '*.jpg'));
patch_size = [100,100];


for i = 1:numel(image_files)
    image = imread(fullfile(image_folder, image_files(i).name));
    image=im2double(im2gray(image));
    n=min(size(image));
    m=11;% support PSF           
    [PSFtilde,~]=psfGauss([m,m],4);
    H_FT=psf2otf(PSFtilde,[n,n]);
    b= real(ifft2(H_FT.*fft2(image)));
    noise = sigma3*randn(n,n);
    bb=b+noise;
    [~, name, ext] = fileparts(image_files(i).name);
    patch_name = sprintf('%s_.jpg', name);
    imwrite(bb,fullfile(patch_folder, patch_name),'jpg');
end