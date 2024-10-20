function [sigma,mu]=meanvariance(M,bb)

A=selezionaPatchManualmente(M,bb);

a=zeros(numel(A),1);

numel(A)

b=a;

for i=1:numel(A)

    a(i)=mean(A{i}(:))

    b(i)=var(A{i}(:));

end

mu=mean(a(:));

sigma=sqrt(mean(b(:)));


end


function patchSelezionate = selezionaPatchManualmente(M,bb)
    % Leggi l'immagine
    immagine = M.*bb;

    % Mostra l'immagine
    imshow(immagine);
    title('Seleziona le patch: premi due volte per terminare');

    % Inizializza una cella per memorizzare le patch selezionate
    patchSelezionate = cell(1, 0);

    % Loop per selezionare le patch
    while true
       % Attendere l'input dell'utente
        h = imrect;
        wait(h);

        % Ottieni la posizione del rettangolo selezionato
        posizione = getPosition(h);
        
        % Arrotonda i valori della posizione al più vicino numero intero
        posizione = round(posizione);
        
        % Ritaglia il segmento dall'immagine originale
        patch = immagine(posizione(2):posizione(2)+posizione(4), posizione(1):posizione(1)+posizione(3));

        % Aggiungi il segmento alla cella dei segmenti ritagliati
        patchSelezionate{end+1} = patch;

        % Aggiungi la patch alla cella delle patch selezionate
        %patchSelezionate{end+1} = patch;

        % Chiedi all'utente se desidera selezionare un'altra patch o terminare
        risposta = questdlg("Vuoi selezionare un' altra patch?', 'Continua', 'Sì', 'No', 'No'");

        if strcmp(risposta, 'No')
            break;
        end
    end
end