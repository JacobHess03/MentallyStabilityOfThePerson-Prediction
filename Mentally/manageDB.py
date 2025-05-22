from DBconnection import db_connection, import_csv_to_table
import os

def main():
    # Parametri di connessione
    db_name = 'DB_Mentally'
    host = 'localhost'
    user = 'root'
    password = 'password'

    # Crea o connetti al database
    conn = db_connection(db_name, host=host, user=user, password=password)

    # Directory contenente i CSV da importare
    csv_dir = os.path.join(os.path.dirname(__file__), 'data')

    # Cicla tutti i file .csv nella cartella
    for filename in os.listdir(csv_dir):
        if filename.lower().endswith('.csv'):
            table_name = os.path.splitext(filename)[0]
            csv_path = os.path.join(csv_dir, filename)

            print(f"Importazione di '{filename}' nella tabella '{table_name}'...")
            import_csv_to_table(conn, csv_path, table_name, if_exists='replace')

    # Chiudi la connessione
    conn.close()
    print("Tutte le tabelle sono state importate e la connessione è chiusa.")

if __name__ == '__main__':
    main()
