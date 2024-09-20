import os
import shutil
import sqlite3
import pandas as pd
import requests

conn = sqlite3.connect("test_db_betano.sqlite")
local_file = ''

db_url = "Falta a url do DB"
local_file = "betano.sqlite"

backup_file = "betano.backup.sqlite"
overwrite = False

if overwrite or not os.path.exists(local_file):
    response = requests.get(db_url)
    response.raise_for_status()  
    with open(local_file, "wb") as f:
        f.write(response.content)
    
    shutil.copy(local_file, backup_file)

def update_dates(file):
    shutil.copy(backup_file, file)
    conn = sqlite3.connect(file)
    cursor = conn.cursor()

    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    ).name.tolist()
    tdf = {}
    for t in tables:
        tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

    example_time = pd.to_datetime(
        tdf["hora_aposta_resultado"]["hora_aposta"].replace("\\N", pd.NaT)
    ).max()
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    time_diff = current_time - example_time

    datetime_columns = ["hora_aposta",
                        "hora_aposta_resultado"]
    
    for column in datetime_columns:
        tdf["historico_apostas"][column] = (
            pd.to_datetime(tdf["historico_apostas"][column].replace("\\N", pd.NaT)) + time_diff
        )

    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    del df
    del tdf
    conn.commit()
    conn.close()

    return file

db = update_dates(local_file)