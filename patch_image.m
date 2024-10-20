clear all 
close all
clc


% Loop attraverso ogni immagine nella cartella
image_folder = 'learning\Sigma1';
patch_folder = 'patch_learning\Sigma1';
image_files = dir(fullfile(image_folder, '*.jpg'));
patch_size = [100,100];


for i = 1:numel(image_files)
    % Leggi l'immagine
    image = imread(fullfile(image_folder, image_files(i).name));
    image=im2double(im2gray(image));
    n=min(size(image));
    image=image(1:n,1:n);
    
    % Loop per estrarre 10 patch
    for j = 1:floor(sqrt(10))
        for k=1:floor(sqrt(10))
        % Calcola la posizione x e y per la patch
        x = (j-1) * patch_size + 1;
        y = (k-1) * patch_size + 1; % aggiorna y se vuoi estrarre patch in verticale invece che in orizzontale
        
        patch = image(y:y+patch_size-1, x:x+patch_size-1);

        % Taglia la patch dall'immagine originale

        % Salva la patch come nuova immagine
        [~, name, ext] = fileparts(image_files(i).name);
        patch_name = sprintf('%s_%d_%d_.jpg', name,j,k);
        imwrite(patch,fullfile(patch_folder, patch_name),'jpg');
        end
    end
end


% Loop attraverso ogni immagine nella cartella
image_folder = 'learning\Sigma2';
patch_folder = 'patch_learning\Sigma2';
image_files = dir(fullfile(image_folder, '*.jpg'));
patch_size = [100,100];


for i = 1:numel(image_files)
    % Leggi l'immagine
    image = imread(fullfile(image_folder, image_files(i).name));
    image=im2double(im2gray(image));
    n=min(size(image));
    image=image(1:n,1:n);
    
    % Loop per estrarre 10 patch
    for j = 1:floor(sqrt(10))
        for k=1:floor(sqrt(10))
        % Calcola la posizione x e y per la patch
        x = (j-1) * patch_size + 1;
        y = (k-1) * patch_size + 1; % aggiorna y se vuoi estrarre patch in verticale invece che in orizzontale
        
        patch = image(y:y+patch_size-1, x:x+patch_size-1);

        % Taglia la patch dall'immagine originale

        % Salva la patch come nuova immagine
        [~, name, ext] = fileparts(image_files(i).name);
        patch_name = sprintf('%s_%d_%d_.jpg', name,j,k);
        imwrite(patch,fullfile(patch_folder, patch_name),'jpg');
        end
    end
end


% Loop attraverso ogni immagine nella cartella
image_folder = 'learning\Sigma3';
patch_folder = 'patch_learning\Sigma3';
image_files = dir(fullfile(image_folder, '*.jpg'));
patch_size = [100,100];


for i = 1:numel(image_files)
    % Leggi l'immagine
    image = imread(fullfile(image_folder, image_files(i).name));
    image=im2double(im2gray(image));
    n=min(size(image));
    image=image(1:n,1:n);
    
    % Loop per estrarre 10 patch
    for j = 1:floor(sqrt(10))
        for k=1:floor(sqrt(10))
        % Calcola la posizione x e y per la patch
        x = (j-1) * patch_size + 1;
        y = (k-1) * patch_size + 1; % aggiorna y se vuoi estrarre patch in verticale invece che in orizzontale
        
        patch = image(y:y+patch_size-1, x:x+patch_size-1);

        % Taglia la patch dall'immagine originale

        % Salva la patch come nuova immagine
        [~, name, ext] = fileparts(image_files(i).name);
        patch_name = sprintf('%s_%d_%d_.jpg', name,j,k);
        imwrite(patch,fullfile(patch_folder, patch_name),'jpg');
        end
    end
end
