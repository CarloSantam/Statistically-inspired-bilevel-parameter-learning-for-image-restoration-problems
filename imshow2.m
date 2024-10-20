function imshow2(X)

%imagesc(X);
imshow(uint8(255*X))
colormap gray
axis square
axis off