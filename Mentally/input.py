import pandas as pd
from preprocessing import preprocess_person_test
import joblib

def gather_input(feature_columns):
    """
    Chiede all'utente di inserire i valori per ciascuna colonna non preprocessata.
    Restituisce un DataFrame con una singola riga comprensivo di 'id' dummy.
    """
    print("Inserisci i seguenti dati:")
    data = {}
    # aggiungiamo un id fisso (dummy) richiesto da preprocess_test
    data['id'] = [0]
    for col in feature_columns:
        val = input(f"- {col}: ")
        # Prova a convertire in numerico, altrimenti lascia stringa
        try:
            val = float(val) if '.' in val else int(val)
        except ValueError:
            pass
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
