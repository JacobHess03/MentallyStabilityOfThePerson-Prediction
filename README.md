## Mentally – Predizione della Depressione

**Descrizione**

Mentally è un progetto di Machine Learning finalizzato a predire se un individuo potrebbe essere affetto da depressione, basandosi su variabili psicologiche e comportamentali.

---

### Caratteristiche principali

1. **Preprocessing modulare**

   * Pulizia dei dati
   * Gestione dei valori mancanti
   * Codifica delle variabili categoriche
   * Normalizzazione e standardizzazione
2. **Modello di classificazione**

   * Algoritmo: XGBoost
   * Tecnica SMOTE per bilanciare le classi
   * Modello pre-addestrato salvato in `best_xgb_clf_smote.pkl`
3. **Script interattivo**

   * `input.py`: raccoglie dati grezzi dall’utente, valida i range numerici e restituisce la predizione
4. **Pipeline completa**

   * Caricamento dati
   * Preprocessing
   * Predizione

---

### Struttura del progetto

```
Mentally/
├── best_xgb_clf_smote.pkl        # Modello XGBoost pre-addestrato
├── input.py                      # Script interattivo per predizione singola
├── main.py                       # Esegue il flusso completo su dataset di test
├── preprocessing.py              # Funzioni di preprocessing per train e test
├── test.csv                      # Dataset di test grezzi
├── submission.csv                # Predizioni su test set (id, Depression)
└── README.md                     # Documentazione di progetto
```

---

### Requisiti di sistema

* Python 3.7 o superiore
* Librerie:

  * pandas
  * scikit-learn
  * xgboost
  * joblib
  * seaborn
  * matplotlib

Installazione dei pacchetti:

```bash
pip install pandas scikit-learn xgboost joblib seaborn matplotlib
```

---

### Utilizzo

#### 1. Predizione interattiva

1. Posizionarsi nella directory del progetto.
2. Eseguire il comando:

   ```bash
   python input.py
   ```
3. Inserire i dati richiesti (i range numerici saranno indicati nel prompt).
4. Il programma restituirà in console l’esito della predizione.

#### 2. Predizione batch

1. Assicurarsi che `test.csv` sia aggiornato nella cartella del progetto.
2. Eseguire:

   ```bash
   python main.py
   ```
3. Il file `submission.csv` verrà creato o aggiornato, contenente:

   * `id`: identificatore di ciascun record
   * `Depression`: predizione (0 = non depressa, 1 = depressa)

---

### Panoramica dei moduli

* **main.py**

  * Carica `test.csv`
  * Applica `preprocess_test`
  * Carica il modello e predice per ogni record
  * Salva o aggiorna `submission.csv`

* **input.py**

  * Raccoglie i dati raw dall’utente con validazione di range
  * Applica `preprocess_test`
  * Carica il modello e restituisce predizione singola in console

* **preprocessing.py**

  * `preprocess_train(df_train)`: gestionde del train set e bilanciamento
  * `preprocess_test(df_test)`: prepare il test set e restituisce ID originali

---

### Dettagli tecnici

* **Codifica**: ordinal encoding per variabili categoriali specifiche
* **Validazione**: prompt di `input.py` include i limiti per le colonne numeriche
* **Persistenza**: modello serializzato con `joblib`

---

### Contributi

Per contribuire:

1. Fork del repository
2. Creazione di un branch dedicato per la feature
3. Invio di una pull request con descrizione delle modifiche

---
*Autore: Giacomo Visciotti-Simone Verrengia-Giuseppe Del Vecchio-Liliana Gilca*
