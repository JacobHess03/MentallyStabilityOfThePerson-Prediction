

# Mentally Stability Of The Person - Depression

## Introduzione

Questo progetto affronta il tema critico della salute mentale attraverso l'analisi di un dataset, con l'obiettivo di sviluppare un modello di Machine Learning capace di predire la probabilità di uno stato depressivo. Implementa una pipeline completa che include fasi di preprocessing avanzato, addestramento di diversi modelli di classificazione, valutazione delle performance e una semplice interfaccia a menu per l'interazione.

## Funzionalità Principali

Il progetto offre le seguenti funzionalità:

* **Esperienza Utente Intuitiva:**
    * Un'interfaccia grafica chiara e semplice per navigare tra le funzionalità offerte.
    * Addestramento dei modelli personalizzabile, con la possibilità di specificare il percorso dei dati.
    * Visualizzazione immediata di grafici esplicativi per comprendere meglio i risultati.
    * Predizione personalizzata della probabilità di depressione, semplicemente inserendo i dati di un utente.
* **Pipeline di Addestramento Modelli:**
    * Addestramento di modelli di classificazione robusti come Logistic Regression, XGBoost e XGBoost con SMOTE.
    * Strategie mirate per gestire il potenziale sbilanciamento delle classi nel dataset (stato depressivo presente vs assente), inclusi `class_weight`, `scale_pos_weight` e l'applicazione di SMOTE tramite `imblearn.pipeline`.
    * Ottimizzazione degli iperparametri del modello XGBoost utilizzando GridSearchCV con cross-validation.
* **Valutazione e Confronto Performance:**
    * Calcolo e visualizzazione delle metriche di valutazione standard: Accuracy, Precision, Recall, e F1-Score.
    * Generazione e visualizzazione comparativa delle Matrici di Confusione per una comprensione approfondita delle performance dei modelli.
* **Predizione su Nuovi Dati:**
    * Applicazione del preprocessing identico a quello del training set per garantire coerenza.
    * Caricamento del modello addestrato ottimizzato (`best_xgb_clf_smote.pkl`).
    * Le previsioni sul dataset di test vengono generate e salvate nel file submission.csv nel formato ID/Predizione, adottando lo standard definito da una competizione su Kaggle che ha ispirato questo progetto.
    * Possibilità di ottenere una predizione per un singolo individuo inserendo i dati manualmente.


## Struttura del Progetto

La struttura delle cartelle e dei file è organizzata come segue:

```
Mentally/
├── `grafici.py`
├── `graphicGui.py`
├── `insertUtente.py`
├── `mainTest.py`
├── `mainTrain.py`
├── `mentallyApp.py`
├── `menu.py`
├── `preprocessing.py`
│
└── data/
    ├── `cleaned_data_for_graphs.csv`
    ├── `cleaned_train.csv`
    ├── `submission.csv`
    ├── `test.csv`
    └── `train.csv`
│
└── modelli/
    ├── `best_xgb_clf_smote.pkl`
    ├── `logistic_regression_balanced.pkl`
    └── `xgb_clf_default_scaled.pkl`
```

### Requisiti

Assicurati di avere **Python 3.6+** installato. Le librerie Python necessarie sono:

* `pandas`
* `numpy`
* `scikit-learn`
* `imbalanced-learn`
* `xgboost`
* `statsmodels`
* `seaborn`
* `matplotlib`
* `joblib`

### Esegui l'applicazione

* Eseguire il file `mentallyApp.py`
* Seleziona l'opzione desiderata digitando il numero corrispondente.
* **Opzione 1 (Addestramento):** Questa è la fase iniziale. Carica `train.csv`, pulisce i dati (`preprocessing.py`), seleziona le feature, addestra e ottimizza i modelli, valuta i risultati e salva i modelli addestrati (`.pkl`) e il dataset pulito in `Mentally/data`. **È indispensabile eseguire questa opzione almeno una volta prima di procedere con le opzioni 2 e 3, poiché queste ultime dipendono dai modelli salvati.**
* **Opzione 2 (Visualizza grafici):** Carica `cleaned_data_for_graphs.csv` e tramite gli specifici bottoni puoi selezionare i vari grafici da visualizzare
* **Opzione 3 (Predizione Singola):** Permette di inserire manualmente i dati di un individuo per ottenere una predizione in tempo reale. Richiama le logiche di preprocessing per un singolo record (`preprocess_person_test` in `preprocessing.py`) e utilizza il modello salvato. **Nota: Richiede la presenza del file `insertUtente.py`.**


*Autore: Giacomo Visciotti-Simone Verrengia-Giuseppe Pio del Vecchio-Liliana Gilca*

