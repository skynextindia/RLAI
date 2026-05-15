
import pandas as pd
import psycopg2
import yaml
import os
from sqlalchemy import create_engine

def load_parquet_to_db(parquet_path, symbol):
    with open("config/base.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    conn_str = config['database']['conn_str']
    # Convert postgresql:// to postgresql+psycopg2:// for sqlalchemy
    alchemy_conn = conn_str.replace("postgresql://", "postgresql+psycopg2://")
    
    print(f"Loading {parquet_path} into DB for symbol {symbol}...")
    df = pd.read_parquet(parquet_path)
    
    # Ensure symbol column exists
    df['symbol'] = symbol
    
    # Map columns to match DB schema if necessary
    # The DB schema from ticks.sql:
    # time, symbol, bid, ask, last, volume, spread, price_delta, volume_delta, time_delta_ms, session
    
    engine = create_engine(alchemy_conn)
    
    try:
        df.to_sql('ticks', engine, if_exists='append', index=False)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parquet_file = "data/historical/BTCUSDm_ticks.parquet"
    if os.path.exists(parquet_file):
        load_parquet_to_db(parquet_file, "BTCUSDm")
    else:
        print(f"File {parquet_file} not found.")
