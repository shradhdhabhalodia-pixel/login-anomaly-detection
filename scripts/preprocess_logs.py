import pandas as pd
import os

INPUT_FILE = "data/raw/logon.csv"
OUTPUT_FILE = "data/processed/login_features.csv"

chunks = []
chunk_size = 100_000   # safe for memory

for chunk in pd.read_csv(INPUT_FILE, chunksize=chunk_size):
    chunk['date'] = pd.to_datetime(chunk['date'])
    chunk['login_hour'] = chunk['date'].dt.hour
    chunk['is_after_hours'] = ((chunk['login_hour'] < 9) | (chunk['login_hour'] > 18)).astype(int)
    chunks.append(chunk[['user', 'login_hour', 'is_after_hours']])

df = pd.concat(chunks)
df.to_csv(OUTPUT_FILE, index=False)

print("Feature file created safely.")
