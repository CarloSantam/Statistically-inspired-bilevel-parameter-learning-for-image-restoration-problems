
function [PSF, center] = psfGauss(dim, s)
if (nargin < 1)
   error('dim must be given.')
end
l = length(dim);
if l == 1
  m = dim;
  n = dim;
else
  m = dim(1);
  n = dim(2);
end
if (nargin < 2)
  s = 2.0;
end
if length(s) == 1
  s = [s,s];
end
%
% Creiamo una griglia per la discretizzare la funzione gaussiana.
%
x = -fix(n/2):ceil(n/2)-1;
y = -fix(m/2):ceil(m/2)-1;
[X,Y] = meshgrid(x,y);
%
% Calcoliamo e normalizziamo la PSF.
%
PSF = exp( -(X.^2)/(2*s(1)^2) - (Y.^2)/(2*s(2)^2) );
PSF = PSF / sum(PSF(:));
%
% Calcoliamo il centro.
%
if nargout == 2
  [mm, nn] = find(PSF == max(PSF(:)));
  center = [mm(1), nn(1)];
end