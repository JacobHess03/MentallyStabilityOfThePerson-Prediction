import pandas as pd
from preprocessing import preprocess_person_test
import joblib

# def gather_input(feature_columns):
#     """
#     Chiede all'utente di inserire i valori per ciascuna colonna non preprocessata.
#     Restituisce un DataFrame con una singola riga comprensivo di 'id' dummy.
#     """
#     print("Inserisci i seguenti dati:")
#     data = {}
#     # aggiungiamo un id fisso (dummy) richiesto da preprocess_test
#     data['id'] = [0]
#     for col in feature_columns:
#         val = input(f"- {col}: ")
#         # Prova a convertire in numerico, altrimenti lascia stringa
#         try:
#             val = float(val) if '.' in val else int(val)
#         except ValueError:
#             pass
#         data[col] = [val]
#     return pd.DataFrame(data)




def gather_input(feature_columns):
    """
    Chiede all'utente di inserire i valori per ciascuna colonna non preprocessata.
    Validazione per colonne con range specifici e indicazione del range nel prompt.
    Restituisce un DataFrame con una singola riga comprensivo di 'id' dummy.
    """
    print("Inserisci i seguenti dati:")
    data = {'id': [0]}

    # Definisci range per alcune colonne
    ranges = {
        'Academic Pressure': (1, 5),
        'Work Pressure': (1, 5),
        'Study Satisfaction': (1, 5),
        'Job Satisfaction': (0, 5),
        'Financial Stress': (1, 5),
        'CGPA': (0.0, 10.0),
        'Sleep Duration': (0.0, 24.0),
        'Work/Study Hours': (0.0, 24.0)
    }

    for col in feature_columns:
        # Prepara prompt con range se disponibile
        if col in ranges:
            lo, hi = ranges[col]
            prompt = f"- {col} (inserisci valore tra {lo} e {hi}): "
        else:
            prompt = f"- {col}: "

        while True:
            raw = input(prompt)
            val = raw
            try:
                if col in ranges:
                    lo, hi = ranges[col]
                    # determina tipo numerico
                    if isinstance(lo, int) and isinstance(hi, int):
                        val_num = int(raw)
                    else:
                        val_num = float(raw)
                    if val_num < lo or val_num > hi:
                        print(f"Valore non valido: inserisci un valore tra {lo} e {hi}.")
                        continue
                    val = val_num
                else:
                    # prova conversione numerica generale
                    val = float(raw) if '.' in raw else int(raw)
            except ValueError:
                # per colonne testuali, accettiamo la stringa raw
                val = raw
            break
        data[col] = [val]

    return pd.DataFrame(data)

def main():
    # Definisci qui le colonne raw (escludendo 'id')
    feature_columns = [
        'Name',
        'Gender',
        'Age',
        'City',
        'Working Professional or Student',
        'Profession',
        'Academic Pressure',
        'Work Pressure',
        'CGPA',
        'Study Satisfaction',
        'Job Satisfaction',
        'Sleep Duration',
        'Dietary Habits',
        'Degree',
        'Have you ever had suicidal thoughts ?',
        'Work/Study Hours',
        'Financial Stress',
        'Family History of Mental Illness'
    ]

    # Raccogli i dati dell'utente
    user_df = gather_input(feature_columns)

    # Applica il preprocessing (ora user_df contiene 'id')
    df_clean = preprocess_person_test(user_df)

    # Carica il modello
    model = joblib.load('best_xgb_clf_smote.pkl')

    # Predizione
    y_pred = model.predict(df_clean)

    # Stampa il risultato per l'utente
    if y_pred[0] == 1:
        print("Risultato: La persona potrebbe essere depressa.")
    else:
        print("Risultato: La persona non sembra depressa.")


if __name__ == '__main__':
    main()
