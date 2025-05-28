import os
import pandas as pd
from mongoengine import connect, DynamicDocument, fields, disconnect


def mongo_connection(db_name: str,
                     host: str = 'localhost',
                     port: int = 27017,
                     username: str = None,
                     password: str = None,
                     authentication_source: str = 'admin'):
    """
    Crea una connessione a MongoDB usando mongoengine.
    """
    disconnect()
    connect(db=db_name,
            host=host,
            port=port,
            username=username,
            password=password,
            authentication_source=authentication_source)
    print(f"Connesso a MongoDB database '{db_name}' su {host}:{port}.")


def import_csv_to_collection(csv_path: str,
                             collection_name: str,
                             db_alias: str = 'default',
                             if_exists: str = 'replace') -> None:
    """
    Importa un file CSV in una collection MongoDB usando mongoengine.
    """
    # Carica CSV
    df = pd.read_csv(csv_path, parse_dates=True)
    # Rinomina colonne riservate per evitare mapping a ObjectId
    rename_map = {}
    for reserved in ['_id', 'id']:
        if reserved in df.columns:
            rename_map[reserved] = f"csv_{reserved}"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Prepara i record come dizionari Python
    def fmt(val):
        if pd.isna(val):
            return None
        if isinstance(val, pd.Timestamp):
            return val.to_pydatetime()
        return val
    records = [{col: fmt(val) for col, val in row.items()} for row in df.to_dict(orient='records')]

    # Definisci dinamicamente un document per la collection
    attrs = {
        'meta': {
            'collection': collection_name,
            'db_alias': db_alias,
            'strict': False
        }
    }
    # Aggiungi campi StringField/IntField etc per ogni colonna
    for col, dtype in df.dtypes.items():
        if pd.api.types.is_integer_dtype(dtype):
            attrs[col] = fields.LongField()
        elif pd.api.types.is_float_dtype(dtype):
            attrs[col] = fields.FloatField()
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            attrs[col] = fields.DateTimeField()
        else:
            attrs[col] = fields.StringField()

    DocumentClass = type(collection_name, (DynamicDocument,), attrs)

    # Gestione if_exists
    if if_exists == 'replace':
        DocumentClass.drop_collection()
    elif if_exists == 'fail':
        if DocumentClass.objects.first():
            raise ValueError(f"Collection '{collection_name}' esiste già e contiene dati.")

    # Inserimento bulk
    if records:
        DocumentClass.objects.insert([DocumentClass(**rec) for rec in records], load_bulk=False)
        print(f"Importati {len(records)} documenti in '{collection_name}'.")
    else:
        print("Nessun documento da importare.")

# Esempio:
mongo_connection('mentally_db')
import_csv_to_collection('Mentally/data/train.csv', 'train_data', if_exists='replace')
