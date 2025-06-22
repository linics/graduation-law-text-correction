import sqlite3
from passlib.hash import pbkdf2_sha256
from typing import Tuple, List, Dict

DB_PATH = "app.db"

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS logs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            action TEXT,
            payload TEXT,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS config(
            key TEXT PRIMARY KEY,
            value TEXT
        )"""
    )
    cur.execute("INSERT OR IGNORE INTO config(key, value) VALUES('alpha','0.4')")
    cur.execute("INSERT OR IGNORE INTO config(key, value) VALUES('beta','0.4')")
    cur.execute("INSERT OR IGNORE INTO config(key, value) VALUES('gamma','0.1')")
    conn.commit()
    conn.close()

def register(username: str, password: str, role: str = "user") -> bool:
    try:
        hashed = pbkdf2_sha256.hash(password)
        conn = get_conn()
        conn.execute(
            "INSERT INTO users(username, password, role) VALUES(?,?,?)",
            (username, hashed, role),
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def verify(username: str, password: str) -> Tuple[bool, str]:
    conn = get_conn()
    cur = conn.execute(
        "SELECT password, role FROM users WHERE username=?", (username,)
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return False, ""
    hashed, role = row
    if pbkdf2_sha256.verify(password, hashed):
        return True, role
    return False, ""

def insert_log(user: str, action: str, payload: str):
    conn = get_conn()
    conn.execute(
        "INSERT INTO logs(user, action, payload) VALUES(?,?,?)",
        (user, action, payload),
    )
    conn.commit()
    conn.close()

def get_logs(offset: int = 0, limit: int = 20) -> List[tuple]:
    conn = get_conn()
    cur = conn.execute(
        "SELECT id, user, action, payload, ts FROM logs ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    )
    rows = cur.fetchall()
    conn.close()
    return rows

def count_logs() -> int:
    conn = get_conn()
    cur = conn.execute("SELECT COUNT(*) FROM logs")
    count = cur.fetchone()[0]
    conn.close()
    return count

def export_all_logs() -> List[tuple]:
    conn = get_conn()
    cur = conn.execute(
        "SELECT id, user, action, payload, ts FROM logs ORDER BY id DESC"
    )
    rows = cur.fetchall()
    conn.close()
    return rows

def load_config() -> Dict[str, float]:
    conn = get_conn()
    cur = conn.execute("SELECT key, value FROM config")
    data = {k: v for k, v in cur.fetchall()}
    conn.close()
    return {
        "alpha": float(data.get("alpha", 0.4)),
        "beta": float(data.get("beta", 0.4)),
        "gamma": float(data.get("gamma", 0.1)),
    }

def save_config(alpha: float, beta: float, gamma: float):
    conn = get_conn()
    conn.execute("REPLACE INTO config(key, value) VALUES('alpha', ?)", (str(alpha),))
    conn.execute("REPLACE INTO config(key, value) VALUES('beta', ?)", (str(beta),))
    conn.execute("REPLACE INTO config(key, value) VALUES('gamma', ?)", (str(gamma),))
    conn.commit()
    conn.close()
