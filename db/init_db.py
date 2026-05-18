import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import DB_PATH

def init():
    schema = (Path(__file__).parent / "schema.sql").read_text()
    con = sqlite3.connect(DB_PATH)
    con.executescript(schema)
    con.commit()
    con.close()
    print(f"Database initialised at {DB_PATH}")

if __name__ == "__main__":
    init()
