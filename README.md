

MentallyStabilityOfThePerson-Predication
Introduzione
Questo progetto mira a esplorare i fattori che possono influenzare lo stato depressivo, analizzando un dataset specifico e sviluppando modelli di Machine Learning per predire la probabilità che un individuo possa soffrire di depressione. Data la crescente importanza della salute mentale, l'obiettivo è fornire uno strumento analitico e predittivo basato sui dati.

Il progetto copre l'intera pipeline di Machine Learning, dalla pulizia e preparazione dei dati all'addestramento e valutazione dei modelli, offrendo anche funzionalità per la predizione su nuovi dati e l'interazione tramite un menu.

Funzionalità Implementate
Preprocessing Dati Robusto:
Gestione avanzata dei valori mancanti con logiche condizionali basate sul ruolo (Student vs Working Professional).
Tecniche di imputazione (mediana, moda) specifiche per tipo di variabile e sottogruppo.
Parsing e standardizzazione di formati dati non uniformi (es. Sleep Duration, Degree).
Codifica di variabili categoriche nominali e ordinali (Genere, Stato Lavorativo/Studentesco, Abitudini Alimentari, Livello di Istruzione raggruppato, Storia Familiare, Regione raggruppata, Gruppo Professionale raggruppato).
Raggruppamento di categorie sparse o non standardizzate (es. diversi titoli di studio in Degree_Group, diverse professioni in Professional_Group, città in Region).
Analisi e selezione delle feature basata su Variance Inflation Factor (VIF) e p-value per mitigare la multicollinearità e identificare variabili significative (applicata sul training set prima dello split).
Esportazione dei dataset pre-processati (cleaned_train.csv, cleaned_test.csv, person_test.csv).
Addestramento e Ottimizzazione Modelli:
Implementazione di diversi algoritmi di classificazione: Logistic Regression e XGBoost.
Strategie per affrontare lo sbilanciamento del dataset (classi di depressione vs non depressione):
Uso di class_weight='balanced' per Logistic Regression.
Calcolo e applicazione di scale_pos_weight per XGBoost.
Integrazione di SMOTE (Synthetic Minority Over-sampling Technique) tramite imblearn.pipeline per creare campioni sintetici della classe minoritaria.
Ottimizzazione degli iperparametri di XGBoost (con pipeline SMOTE) utilizzando GridSearchCV e cross-validation (CV=5) con metrica di scoring F1-Score (particolarmente rilevante per dataset sbilanciati).
Valutazione Performance:
Calcolo di metriche di classificazione standard: Accuracy, Precision, Recall, F1-Score.
Visualizzazione grafica comparativa delle matrici di confusione per i modelli valutati.
Visualizzazione grafica comparativa delle metriche chiave (Accuracy, Precision, Recall, F1-Score).
Predizione:
Predizione sullo standard test set (test.csv) e generazione di un file di submission (submission.csv) nel formato richiesto (ID e Predizione).
Predizione dello stato depressivo per un singolo individuo basata su input manuale (richiede insertUtente.py).
Interfaccia Utente Semplice:
Menu testuale interattivo (main.py) per navigare tra le principali funzionalità del progetto (addestramento, test, predizione singola, visualizzazioni).
Struttura del Progetto
.
├── Mentally/
│   ├── train.csv                       # Dataset di training originale (richiesto per esecuzione 1)
│   ├── test.csv                        # Dataset di test originale (richiesto per esecuzione 2)
│   ├── submission.csv                  # Output: File CSV con le previsioni sul test set (generato da 2)
│   ├── cleaned_train.csv               # Output: Training set dopo preprocessing (generato da 1)
│   ├── cleaned_test.csv                # Output: Test set dopo preprocessing (generato da 2)
│   ├── person_test.csv                 # Output: File CSV temporaneo per input manuale (generato da 3)
│   ├── best_xgb_clf_smote.pkl          # Output: Modello XGBoost ottimizzato con SMOTE (generato da 1)
│   ├── logistic_regression_smote.pkl   # Output: Modello Logistic Regression bilanciato (generato da 1)
│   └── xgb_clf_default_smote.pkl       # Output: Modello XGBoost default con pesi (generato da 1)
├── main.py                             # Script principale con menu interattivo
├── mainTrain.py                        # Logica per addestramento, valutazione e salvataggio modelli
├── mainTest.py                         # Logica per predizione su test set e generazione submission
├── preprocessing.py                    # Tutte le funzioni per la pulizia e trasformazione dei dati
├── insertUtente.py                     # Gestione input manuale utente e predizione singola (CODICE NON FORNITO)
└── grafici.py                          # Funzioni per visualizzazioni grafiche (CODICE NON FORNITO)
Requisiti di Sistema
Assicurati di avere installato Python 3.6 o superiore. Le dipendenze principali sono:

pandas (per la manipolazione dei dati)
numpy (per operazioni numeriche)
scikit-learn (per split dati, modelli, metriche, encoding)
imbalanced-learn (per SMOTE e pipeline)
xgboost (per il modello XGBoost)
statsmodels (per VIF e p-value)
seaborn e matplotlib (per le visualizzazioni)
joblib (per il salvataggio/caricamento dei modelli)
re (built-in, per regex nel preprocessing)
Puoi installare la maggior parte delle dipendenze tramite pip:

Bash

pip install pandas numpy scikit-learn imbalanced-learn xgboost statsmodels seaborn matplotlib joblib
Installazione
Scarica o clona i file del progetto nella tua directory locale.
Crea una sottocartella chiamata Mentally/ nella directory principale del progetto.
Posiziona i file train.csv e test.csv (il dataset su cui lavorare) all'interno della cartella Mentally/.
Utilizzo
Esegui il programma principale dal terminale:

Bash

python main.py
Verrà visualizzato il menu interattivo:

Benvenuto nel menu principale!
1. Analisi, preprocessing, addestramento e previsione su file CSV di training
2. Analisi, preprocessing, previsione su test e generazione submission
3. Predizione dello stato depressivo di una persona (inserimento manuale)
4. Visualizza grafici
5. Esci
Opzione 1: Esegue l'intera pipeline di addestramento. Carica train.csv, applica il preprocessing, seleziona le feature, splitta in train/test interni per la validazione, addestra i modelli (LogReg, XGBoost, XGBoost con SMOTE + GridSearch), valuta le performance visualizzando metriche e matrici di confusione, e salva i modelli addestrati e il dataset pulito nella cartella Mentally/. Questa opzione deve essere eseguita almeno una volta prima delle opzioni 2 e 3.
Opzione 2: Esegue la pipeline di test/submission. Carica test.csv, applica lo stesso preprocessing definito in preprocessing.py, carica il modello salvato best_xgb_clf_smote.pkl (addestrato nell'opzione 1), effettua le predizioni sul test set e salva il risultato nel file Mentally/submission.csv.
Opzione 3: Avvia l'interfaccia per l'inserimento manuale dei dati di un singolo utente (implementata in insertUtente.py). I dati inseriti vengono pre-processati (usando preprocess_person_test da preprocessing.py), e il modello addestrato (caricato da .pkl) viene utilizzato per fornire una predizione. Richiede che il file insertUtente.py sia presente e che l'opzione 1 sia stata eseguita.
Opzione 4: Accede al menu delle visualizzazioni (implementato in grafici.py). Permette di esplorare grafici sui dati o sui risultati. Richiede che il file grafici.py sia presente.
Opzione 5: Esci dal programma.
Descrizione Dettagliata dei File
main.py:
Il punto di avvio dell'applicazione.
Definisce la funzione menu() che presenta le opzioni all'utente.
Importa e chiama funzioni da mainTrain.py, mainTest.py, insertUtente.py, e grafici.py in base alla scelta dell'utente.
Contiene il blocco if __name__ == "__main__": per garantire l'esecuzione del menu all'avvio dello script.
mainTrain.py:
Contiene la funzione train() che orchestra il processo di addestramento.
Carica il dataset di training (Mentally/train.csv).
Chiama preprocess_train da preprocessing.py.
Definisce le feature X e il target y.
Applica la selezione feature elimina_variabili_vif_pvalue da preprocessing.py sulle feature.
Splitta il dataset in train e test set interni (X_train, X_test, y_train, y_test) usando train_test_split con stratify per mantenere la distribuzione delle classi.
Inizializza e addestra i modelli: Logistic Regression (bilanciato), XGBoost (con scale_pos_weight), e una Pipeline SMOTE + XGBoost.
Configura e esegue GridSearchCV sulla pipeline SMOTE + XGBoost per trovare la combinazione ottimale di n_estimators, max_depth, learning_rate, e scale_pos_weight basata sul F1-Score.
Salva i modelli addestrati (best_xgb_clf_smote.pkl, logistic_regression_smote.pkl, xgb_clf_default_smote.pkl) usando joblib.
Genera predizioni sui test set interni per la valutazione.
Utilizza le funzioni plot_combined_confusion_matrices e plot_metrics_comparison (definite nello stesso file) per visualizzare i risultati della valutazione.
Contiene le funzioni plot_combined_confusion_matrices e plot_metrics_comparison per la visualizzazione comparativa delle performance dei modelli sui dati di test interni.
mainTest.py:
Contiene la funzione test().
Carica il dataset di test (Mentally/test.csv).
Chiama preprocess_test da preprocessing.py, ottenendo il DataFrame pulito (df_clean) e gli ID originali (test_ids).
Carica il modello addestrato ottimizzato (best_xgb_clf_smote.pkl) usando joblib.
Effettua le predizioni (y_pred_class) sul DataFrame di test pulito.
Crea un DataFrame submission con le colonne 'id' (dagli test_ids allineati) e 'Depression' (dalle predizioni).
Salva il DataFrame submission nel file Mentally/submission.csv.
preprocessing.py:
Contiene un set completo di funzioni per la preparazione dei dati, riutilizzate sia per il training che per il testing e l'input manuale.
map_sleep_duration(duration_str): Converte le varie rappresentazioni testuali della durata del sonno in valori numerici (gestisce intervalli, "less than", "more than", valori anomali).
elimina_variabili_vif_pvalue(X, y, vif_threshold, pvalue_threshold): Implementa un processo iterativo per rimuovere le feature che presentano sia alta multicollinearità (VIF > threshold) che bassa significatività statistica (p-value > threshold) in un modello OLS.
preprocess_train(df): Applica le logiche di imputazione specifiche per train/professionista, gestisce i rimanenti NaN, applica le codifiche (Gender, Working Status, Sleep Duration), pulisce e codifica Degree in Degree_Group_Encoded (ordinale), pulisce, gestisce valori non validi e codifica Dietary Habits (ordinale), codifica Suicidal Thoughts e Family History (binary), raggruppa e codifica Profession in Professional_Group_Encoded (label encoding), raggruppa e codifica City in Region_Encoded (label encoding), elimina colonne originali non più necessarie (City, Name, id, Profession, Degree, Degree_Group, Professional_Group, Region), rinomina e droppa la colonna Have you ever had suicidal thoughts ? (rinominata in SuicidalThoughts e poi rimossa), e salva il risultato in cleaned_train.csv.
preprocess_test(df): Simile a preprocess_train ma adatta per il test set. Gestisce l'imputazione, le codifiche e i raggruppamenti esattamente come nel train set per garantire consistenza. Restituisce il DataFrame pulito e gli ID originali test_ids (fondamentali per la submission). Salva il risultato in cleaned_test.csv.
preprocess_person_test(df): Simile alle funzioni preprocess_train e preprocess_test ma pensata per processare un singolo record (proveniente dall'input manuale). Applica le stesse trasformazioni di imputazione e codifica per rendere il formato compatibile con quello atteso dal modello addestrato. Salva il record processato in person_test.csv.
insertUtente.py (CODICE NON FORNITO):
Basato sull'importazione, si presume contenga la funzione insert_data().
Questa funzione dovrebbe guidare l'utente attraverso l'inserimento dei valori per ogni feature necessaria alla predizione (probabilmente le stesse feature che rimangono dopo il preprocessing e la selezione).
Probabilmente crea un DataFrame pandas con la singola riga di input e chiama preprocess_person_test da preprocessing.py.
Successivamente, carica il modello addestrato e usa la riga processata per fare una predizione.
Infine, presenta la predizione all'utente.
grafici.py (CODICE NON FORNITO):
Basato sull'importazione, si presume contenga la funzione menu_visualizzazioni() e altre funzioni di supporto.
Potrebbe offrire opzioni per visualizzare distribuzioni di variabili, relazioni tra feature e target, o visualizzazioni dei risultati del modello (oltre quelle già presenti in mainTrain.py).
Utilizzerebbe le librerie matplotlib e seaborn
