import mysql.connector
import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype, is_datetime64_any_dtype


def db_connection(db_name: str,
                  host: str = "localhost",
                  user: str = "root",
                  password: str = "password") -> mysql.connector.connection_cext.CMySQLConnection:
    """
    Crea e/o connette a un database MySQL.

    Args:
        db_name (str): Nome del database.
        host (str): Host del server MySQL.
        user (str): Utente MySQL.
        password (str): Password MySQL.

    Returns:
        mysql.connector.connection_cext.CMySQLConnection: Oggetto di connessione al database specificato.
    """
    # Connessione iniziale per creare il database se non esiste
    conn = mysql.connector.connect(host=host,
                                   user=user,
                                   password=password)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`;")
    cursor.close()
    conn.close()

    # Connessione al database specifico
    myDB = mysql.connector.connect(host=host,
                                   user=user,
                                   password=password,
                                   database=db_name)
    print(f"Connesso al database '{db_name}' con successo.")
    return myDB


def infer_mysql_type(series: pd.Series) -> str:
    """
    In base al dtype di pandas, ritorna il tipo MySQL appropriato.
    """
    if is_integer_dtype(series):
        return "BIGINT"
    if is_float_dtype(series):
        return "DOUBLE"
    if is_datetime64_any_dtype(series):
        return "DATETIME"
    # default a TEXT per oggetti/stringhe
    return "TEXT"


def import_csv_to_table(conn: mysql.connector.connection_cext.CMySQLConnection,
                        csv_path: str,
                        table_name: str,
                        if_exists: str = "replace") -> None:
    """
    Importa un file CSV in una tabella MySQL con tipi colonna inferiti.

    Args:
        conn: Connessione MySQL già aperta (ritornata da db_connection).
        csv_path (str): Percorso al file CSV.
        table_name (str): Nome della tabella di destinazione.
        if_exists (str): Strategia se la tabella esiste: 'replace', 'append', 'fail'.
    """
    df = pd.read_csv(csv_path, parse_dates=True)
    cursor = conn.cursor()

    # Se richiesto, elimina e ricrea la tabella
    if if_exists == "replace":
        cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`;")
    elif if_exists == "fail":
        cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
        if cursor.fetchone():
            raise ValueError(f"La tabella '{table_name}' esiste già.")

    # Definizione dinamica della tabella in base ai dtypes
    col_defs = []
    for col in df.columns:
        dtype = infer_mysql_type(df[col])
        col_defs.append(f"`{col}` {dtype}")
    schema = ", ".join(col_defs)
    cursor.execute(f"CREATE TABLE IF NOT EXISTS `{table_name}` ({schema});")

    # Costruzione statement di INSERT
    placeholders = ", ".join(["%s"] * len(df.columns))
    cols_backticked = ", ".join([f"`{col}`" for col in df.columns])
    insert_sql = f"INSERT INTO `{table_name}` ({cols_backticked}) VALUES ({placeholders});"

    # Prepara i dati, convertendo i timestamp in stringhe se necessario
    def fmt(val):
        if pd.isna(val):
            return None
        if isinstance(val, pd.Timestamp):
            return val.to_pydatetime().strftime('%Y-%m-%d %H:%M:%S')
        return val

    data = [tuple(fmt(v) for v in row) for row in df.itertuples(index=False, name=None)]

    if if_exists == "append" or if_exists == "replace":
        cursor.executemany(insert_sql, data)
        conn.commit()
        print(f"Importati {len(data)} record in `{table_name}`.")

    cursor.close()
