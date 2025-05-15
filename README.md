

# Mentally Stability Of The Person-Predication

## Introduzione

Questo progetto affronta il tema critico della salute mentale attraverso l'analisi di un dataset, con l'obiettivo di sviluppare un modello di Machine Learning capace di predire la probabilità di uno stato depressivo. Implementa una pipeline completa che include fasi di preprocessing avanzato, addestramento di diversi modelli di classificazione, valutazione delle performance e una semplice interfaccia a menu per l'interazione.

## Funzionalità Principali

Il progetto offre le seguenti funzionalità:

* **Preprocessing Dati Avanzato:**
    * Gestione sofisticata dei valori mancanti tramite tecniche di imputazione (mediana, moda) con logiche differenziate per sottogruppi specifici (Studenti vs Working Professionals).
    * Normalizzazione e standardizzazione di dati testuali e categorici non uniformi (es. durate del sonno, titoli di studio).
    * Codifica di variabili categoriche (binarie, ordinali, nominali) utilizzando mappature personalizzate, OrdinalEncoder e LabelEncoder.
    * Raggruppamento intelligente di categorie con alta cardinalità (es. Professioni, Città) in gruppi più gestibili (Professional Group, Region, Degree Group).
    * Selezione automatica delle feature basata su test statistici (VIF e p-value) per ridurre la multicollinearità e migliorare la stabilità del modello (applicata sul training set prima dello split).
* **Pipeline di Addestramento Modelli:**
    * Addestramento di modelli di classificazione robusti come Logistic Regression, XGBoost e XGBoost con SMOTE.
    * Strategie mirate per gestire il potenziale sbilanciamento delle classi nel dataset (stato depressivo presente vs assente), inclusi `class_weight`, `scale_pos_weight` e l'applicazione di SMOTE tramite `imblearn.pipeline`.
    * Ottimizzazione degli iperparametri del modello XGBoost utilizzando GridSearchCV con cross-validation (5 fold) focalizzata sull'ottimizzazione del F1-Score.
* **Valutazione e Confronto Performance:**
    * Calcolo e visualizzazione delle metriche di valutazione standard: Accuracy, Precision, Recall, e F1-Score.
    * Generazione e visualizzazione comparativa delle Matrici di Confusione per una comprensione approfondita delle performance dei modelli.
* **Predizione su Nuovi Dati:**
    * Applicazione del preprocessing identico a quello del training set per garantire coerenza.
    * Caricamento del modello addestrato ottimizzato (`best_xgb_clf_smote.pkl`).
    * Generazione di previsioni sul dataset di test e creazione di un file `submission.csv` nel formato standard ID/Predizione.
    * Possibilità di ottenere una predizione per un singolo individuo inserendo i dati manualmente (richiede script dedicato).
* **Interfaccia Utente Interattiva:**
    * Menu testuale semplice per lanciare le varie fasi della pipeline (Addestramento, Test, Predizione Singola).
    * Integrazione con uno script esterno per visualizzazioni aggiuntive (richiede script dedicato).

## Struttura del Progetto

La struttura delle cartelle e dei file è organizzata come segue:

```
**Mentally/**
├── `grafici.py`
├── `graphicGui.py`
├── `insertUtente.py`
├── `mentallyApp.py`
├── `menu.py`
├── `preprocessing.py`
│
└── **data/**
    ├── `cleaned_data_for_graphs.csv`
    ├── `cleaned_train.csv`
    ├── `submission.csv`
    ├── `test.csv`
    └── `train.csv`
│
└── **modelli/**
    ├── `best_xgb_clf_smote.pkl`
    ├── `logistic_regression_balanced.pkl`
    └── `xgb_clf_default_scaled.pkl`
```

## Requisiti di Sistema e Installazione

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

Puoi installarle tutte tramite pip:

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost statsmodels seaborn matplotlib joblib
```

Seleziona l'opzione desiderata digitando il numero corrispondente.

* **Opzione 1 (Addestramento):** Questa è la fase iniziale. Carica `train.csv`, pulisce i dati (`preprocessing.py`), seleziona le feature, addestra e ottimizza i modelli, valuta i risultati (con output a console e grafici delle matrici/metriche) e salva i modelli addestrati (`.pkl`) e il dataset pulito in `Mentally/data`. **È indispensabile eseguire questa opzione almeno una volta prima di procedere con le opzioni 2 e 3, poiché queste ultime dipendono dai modelli salvati.**
* **Opzione 2 (Visualizza grafici):** Carica `cleaned_data_for_graphs.csv` e tramite gli specifici bottoni puoi selezionare i vari grafici da visualizzare
* **Opzione 3 (Predizione Singola):** Permette di inserire manualmente i dati di un individuo per ottenere una predizione in tempo reale. Richiama le logiche di preprocessing per un singolo record (`preprocess_person_test` in `preprocessing.py`) e utilizza il modello salvato. **Nota: Richiede la presenza del file `insertUtente.py`.**


*Autore: Giacomo Visciotti-Simone Verrengia-Giuseppe Pio del Vecchio-Liliana Gilca*

