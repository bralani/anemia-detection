# Repository anemia-detection

MIT License
Copyright (c) [2023] [Balice Matteo]

Benvenuti nella repository di anemia-detection! Questa repository contiene codice e risorse correlati per il processo di individuazione dell'anemia attraverso le immagini oculari.

## Contenuto della Repository

La repository è organizzata in tre cartelle principali:

1. **database_sclere**: Questa cartella contiene il dataset di immagini oculari utilizzato per l'addestramento e la valutazione dei modelli di segmentazione della sclera. Le immagini "RAW" sono 218 e organizzate in sottocartelle per facilitare l'accesso e la gestione dei dati.

2. **sclera_segmentation**: In questa cartella troverai il codice e gli script necessari per eseguire la segmentazione della sclera.

3. **entire_workflow.ipynb**: contiene il workflow intero del progetto unito in un unico file (dalla segmentazione della sclera alla predizione finale).

4. **generate_vessel_masks.ipynb** contiene il codice per la segmentazione delle vene sclerali in immagini degli occhi utilizzando un modello di rete neurale convoluzionale (U-Net). Dopo una fase di preelaborazione delle immagini il codice esegue il loop su un DataFrame (df) contenente informazioni sui pazienti. Per ogni riga del DataFrame, viene eseguita la segmentazione delle vene sclerali sull'immagine corrispondente utilizzando il modello U-Net preaddestrato.

5. **density_evaluation.ipynb** contiente uno script che definisce una funzione che calcola e stampa diversi punteggi di valutazione per un modello di classificazione. Sono calcolati i punteggi di: f1, f2, precision, recall, accuracy e roc_auc.

6. **preprocessing.ipynb** contiene uno script che definisce alcune funzioni di preelaborazione delle immagini e poi applica queste funzioni a un set di immagini di addestramento.

6. **sclera_segmentation.ipynb** mostra passo dopo passo la segmentazione automatica della sclera.

## Installazione e Utilizzo

Per iniziare, segui queste istruzioni per l'installazione e l'utilizzo del codice e delle risorse nella repository:

1. Clona questa repository sul tuo sistema locale utilizzando il comando:
```git clone https://github.com/bralani/anemia-detection.git```

2. Seguire il workflow del file entire_workflow.ipynb

## Licenza

Si prega di fare riferimento al file [LICENSE](LICENSE) per ulteriori informazioni sui diritti e le restrizioni legati all'uso, alla modifica e alla distribuzione di questo software.


## Supporto

Per qualsiasi domanda o problema, puoi aprire una issue nella sezione Issue di questa repository. Faremo del nostro meglio per rispondere nel minor tempo possibile.
