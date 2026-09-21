"""
db.py
SQLite-backed storage for the login/signup + KYC gating system.

IMPORTANT (mention this in your report): this is a MOCK/DEMO KYC flow for
academic project purposes. It does NOT perform real government ID
verification (no UIDAI/Aadhaar API integration). Never collect or store
real Aadhaar numbers or other real government ID data with this code —
use only fake/sample data for demos. In a real production system, ID data
must be encrypted at rest, access-logged, and handled per UIDAI/DPDP Act
regulations.

Schema:
    users(id, username, password_hash, is_admin, created_at)
    kyc(user_id, full_name, id_number, dob, address, status, flagged_count,
        submitted_at, decided_at)

KYC status values: 'not_submitted', 'pending', 'approved', 'rejected', 'cancelled'
"""

import sqlite3
import hashlib
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app_data.db')

FLAG_THRESHOLD = 5  # number of flagged comments before auto-cancelling KYC


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS kyc (
            user_id INTEGER PRIMARY KEY,
            full_name TEXT,
            id_number TEXT,
            dob TEXT,
            address TEXT,
            status TEXT DEFAULT 'not_submitted',
            flagged_count INTEGER DEFAULT 0,
            submitted_at TEXT,
            decided_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()

    # Seed a default admin account if none exists (username: admin / password: admin123)
    # CHANGE THIS before showing to anyone outside your immediate demo.
    cur.execute("SELECT COUNT(*) as c FROM users WHERE is_admin = 1")
    if cur.fetchone()['c'] == 0:
        cur.execute(
            "INSERT INTO users (username, password_hash, is_admin, created_at) VALUES (?, ?, 1, ?)",
            ("admin", hash_password("admin123"), datetime.now().isoformat())
        )
        conn.commit()

    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username: str, password: str) -> tuple[bool, str]:
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, is_admin, created_at) VALUES (?, ?, 0, ?)",
            (username, hash_password(password), datetime.now().isoformat())
        )
        user_id = cur.lastrowid
        cur.execute(
            "INSERT INTO kyc (user_id, status) VALUES (?, 'not_submitted')",
            (user_id,)
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()


def verify_login(username: str, password: str):
    """Returns the user row (as dict) if valid, else None."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if row and row['password_hash'] == hash_password(password):
        return dict(row)
    return None


def get_kyc_status(user_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM kyc WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def submit_kyc(user_id: int, full_name: str, id_number: str, dob: str, address: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        UPDATE kyc SET full_name=?, id_number=?, dob=?, address=?,
        status='pending', submitted_at=? WHERE user_id=?
    """, (full_name, id_number, str(dob), address, datetime.now().isoformat(), user_id))
    conn.commit()
    conn.close()


def get_pending_kyc_list():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT kyc.*, users.username FROM kyc
        JOIN users ON users.id = kyc.user_id
        WHERE kyc.status = 'pending'
        ORDER BY kyc.submitted_at ASC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_all_users_with_kyc():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT kyc.*, users.username, users.is_admin FROM kyc
        JOIN users ON users.id = kyc.user_id
        WHERE users.is_admin = 0
        ORDER BY kyc.submitted_at DESC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def decide_kyc(user_id: int, approve: bool):
    conn = get_conn()
    cur = conn.cursor()
    new_status = 'approved' if approve else 'rejected'
    cur.execute(
        "UPDATE kyc SET status=?, decided_at=? WHERE user_id=?",
        (new_status, datetime.now().isoformat(), user_id)
    )
    conn.commit()
    conn.close()


def increment_flag_count(user_id: int) -> int:
    """
    Increments the flagged-comment counter for a user. If it reaches
    FLAG_THRESHOLD, automatically cancels their KYC (revoking access).
    Returns the new flagged_count.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE kyc SET flagged_count = flagged_count + 1 WHERE user_id=?", (user_id,))
    cur.execute("SELECT flagged_count FROM kyc WHERE user_id=?", (user_id,))
    new_count = cur.fetchone()['flagged_count']

    if new_count >= FLAG_THRESHOLD:
        cur.execute(
            "UPDATE kyc SET status='cancelled', decided_at=? WHERE user_id=?",
            (datetime.now().isoformat(), user_id)
        )

    conn.commit()
    conn.close()
    return new_count


def reinstate_kyc(user_id: int):
    """Admin action: manually reinstate a cancelled user (resets flag count)."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE kyc SET status='approved', flagged_count=0, decided_at=? WHERE user_id=?",
        (datetime.now().isoformat(), user_id)
    )
    conn.commit()
    conn.close()
