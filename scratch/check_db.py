
import psycopg2
import yaml

def check_db():
    with open("config/base.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    conn_str = config['database']['conn_str']
    try:
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor()
        cur.execute("SELECT symbol, count(*) FROM ticks GROUP BY symbol;")
        rows = cur.fetchall()
        for row in rows:
            print(f"Symbol: {row[0]}, Count: {row[1]}")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_db()
