Mentally - Predizione della Depressione
Descrizione del Progetto
Il progetto Mentally è un'applicazione di machine learning progettata per predire se un individuo potrebbe essere affetto da depressione. Utilizzando un dataset con variabili psicologiche e comportamentali, il progetto costruisce un modello di classificazione che può prevedere l'esito sulla base delle caratteristiche dell'individuo. Il modello utilizza XGBoost, una potente libreria di gradient boosting, e applica tecniche di SMOTE per affrontare il problema dello sbilanciamento delle classi nel dataset.

Il flusso di lavoro del progetto include:

Caricamento dei dati di test.

Preprocessing dei dati (pulizia e preparazione).

Predizione della depressione utilizzando un modello pre-addestrato.

Creazione e salvataggio di un file di submission contenente le previsioni.

Funzionalità principali
1. Caricamento dei dati di test
Il file di test, contenente le osservazioni su cui fare previsioni, viene caricato tramite il modulo pandas. Si assume che il file sia in formato CSV e contenga tutte le variabili tranne la colonna target Depression (che non è presente nel set di test).

python
Copia
Modifica
test = pd.read_csv(r'Mentally/test.csv')
2. Preprocessing dei dati
Una volta caricato il dataset, il passo successivo è la pulizia e la trasformazione dei dati. La funzione preprocess_test, importata da un modulo separato preprocessing.py, esegue questo processo.

Dettagli del preprocessing:
Gestione dei valori mancanti: Rimuove o sostituisce i dati mancanti (ad esempio, con la media o la mediana).

Codifica delle variabili categoriche: Trasforma le variabili non numeriche in variabili numeriche utilizzabili dal modello di machine learning.

Normalizzazione o standardizzazione: Può essere applicata per migliorare la performance del modello.

La funzione restituisce due valori:

df_clean: il dataset pulito e pronto per il modello.

test_ids: gli ID delle osservazioni, che saranno usati per identificare le predizioni nel file di output.

python
Copia
Modifica
df_clean, test_ids = preprocess_test(test)
3. Caricamento del modello pre-addestrato
Il modello di machine learning utilizzato è un modello XGBoost addestrato precedentemente. Questo modello è stato salvato come file .pkl per essere caricato e riutilizzato senza la necessità di riaddestrarlo.

python
Copia
Modifica
model = joblib.load('best_xgb_clf_smote.pkl')
Il modello è stato addestrato su un dataset con caratteristiche psicologiche e comportamentali e la colonna target Depression, e ha utilizzato una tecnica di SMOTE (Synthetic Minority Over-sampling Technique) per bilanciare le classi nel caso in cui i dati siano sbilanciati.

4. Predizione sui dati di test
Una volta che il modello è caricato, viene usato per fare previsioni sui dati di test (memorizzati in df_clean). Il modello fornisce una predizione binaria (0 o 1) per ogni osservazione, dove:

1 indica che l'individuo è previsto essere affetto da depressione.

0 indica che l'individuo non è previsto essere affetto da depressione.

python
Copia
Modifica
y_pred_class = model.predict(X_test)
5. Creazione del file di submission
Dopo aver ottenuto le predizioni, il progetto crea un DataFrame che contiene due colonne:

id: l'ID di ciascun record nei dati di test, utile per associare le predizioni a specifici individui.

Depression: le predizioni del modello (0 o 1).

Il DataFrame viene quindi esportato come file CSV che può essere inviato per la valutazione del modello (ad esempio, in una competizione di machine learning o per l'analisi dei risultati).

python
Copia
Modifica
submission = pd.DataFrame({
    'id': test_ids,
    'Depression': y_pred_class
})

submission.to_csv('Mentally/submission.csv', index=False)
Alla fine, viene stampato un messaggio per confermare che il file è stato creato correttamente:

python
Copia
Modifica
print("Submission salvata su 'Mentally/submission.csv'")
6. Generazione del file di output
Il file submission.csv contiene le predizioni del modello per ogni record di test. Questo file può essere utilizzato per ulteriori analisi o per inviare i risultati in una competizione.

Struttura del progetto
Il progetto è suddiviso in più moduli per favorire la modularità e la manutenzione. La struttura dei file è la seguente:

bash
Copia
Modifica
Mentally/
├── best_xgb_clf_smote.pkl        # Modello pre-addestrato XGBoost
├── main.py                       # Script principale per la predizione
├── preprocessing.py              # Funzioni per il preprocessing dei dati
├── test.csv                      # Dataset di test
├── submission.csv                # File di output con le predizioni
main.py
Il file principale che esegue il flusso di lavoro completo del progetto. Esso:

Carica i dati di test.

Pulisce e prepara i dati.

Carica il modello XGBoost.

Effettua la predizione sui dati di test.

Crea e salva il file di submission.

preprocessing.py
Contiene la funzione preprocess_test, che si occupa di:

Gestire i valori mancanti nei dati.

Codificare le variabili categoriche.

Eseguire eventuali altre trasformazioni sui dati (normalizzazione, gestione delle anomalie, ecc.).

best_xgb_clf_smote.pkl
Il modello XGBoost pre-addestrato, salvato come file .pkl. Questo file contiene il modello di classificazione che è stato addestrato per prevedere se una persona è affetta da depressione basandosi sulle sue caratteristiche.

test.csv
Il dataset di test contiene i dati sulle caratteristiche degli individui sui quali vengono fatte le predizioni. Non include la colonna target Depression, che è presente solo nel set di addestramento.

submission.csv
Il file CSV che contiene:

id: gli ID delle osservazioni di test.

Depression: le predizioni del modello (0 o 1).

Come Eseguire il Progetto
Per eseguire il progetto, assicurati di avere i seguenti prerequisiti:

Installa le librerie richieste:

bash
Copia
Modifica
pip install pandas xgboost joblib scikit-learn
Preparare il file di test:

Assicurati che il file test.csv sia presente nella stessa directory del progetto o modifica il percorso nel codice.

Esegui il file principale:

Una volta configurato il progetto, puoi eseguire il flusso di lavoro semplicemente eseguendo il file main.py:

bash
Copia
Modifica
python main.py
Verifica il file di output:

Dopo aver eseguito il codice, il file submission.csv verrà creato nella cartella Mentally/, contenente le predizioni del modello.

Requisiti
Per eseguire correttamente il progetto, è necessario avere le seguenti librerie Python:

pandas per la manipolazione dei dati.

xgboost per l'addestramento del modello.

joblib per il salvataggio e il caricamento del modello.

scikit-learn per funzioni di preprocessing e machine learning.

Puoi installarle tutte insieme con:

bash
Copia
Modifica
pip install pandas xgboost joblib scikit-learn
