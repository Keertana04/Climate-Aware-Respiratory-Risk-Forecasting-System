import sqlite3
import json
import pathlib

DB_DIR = pathlib.Path(__file__).parent / "users_db"
DB_FILE = DB_DIR / "users.db"

def _get_connection():
    # Make sure DB dir exists
    DB_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    # Ensure table exists if DB is completely fresh
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            uid TEXT PRIMARY KEY,
            email TEXT,
            phone TEXT,
            name TEXT,
            age TEXT,
            gender TEXT,
            smoker TEXT,
            chronic_conditions TEXT,
            temp_conditions TEXT
        )
    ''')
    conn.commit()
    return conn

def _row_to_dict(row):
    d = dict(row)
    # Parse json lists back
    try:
        d['chronic_conditions'] = json.loads(d['chronic_conditions']) if d.get('chronic_conditions') else []
    except:
        d['chronic_conditions'] = []
    try:
        d['temp_conditions'] = json.loads(d['temp_conditions']) if d.get('temp_conditions') else []
    except:
        d['temp_conditions'] = []
    
    # Backwards compatibility injection for API responses
    d['contact'] = d.get('phone') or d.get('email') or d.get('uid')
    return d

def get_user(email, phone):
    e_str = str(email).lower().strip() if email else ""
    p_str = str(phone).strip() if phone else ""
    
    conn = _get_connection()
    c = conn.cursor()
    
    query = '''
        SELECT * FROM users 
        WHERE (email = ? AND email != '') 
           OR (phone = ? AND phone != '')
           OR (uid = ? AND uid != '') 
           OR (uid = ? AND uid != '')
    '''
    c.execute(query, (e_str, p_str, e_str, p_str))
    row = c.fetchone()
    conn.close()
    
    if row:
        return _row_to_dict(row)
    return None

def _is_email_taken(email, conn):
    """Returns True if the email is already registered (regardless of phone)."""
    if not email:
        return False
    c = conn.cursor()
    c.execute("SELECT uid FROM users WHERE email = ? AND email != ''", (email,))
    return c.fetchone() is not None

def _is_phone_taken(phone, conn):
    """Returns True if the phone is already registered (regardless of email)."""
    if not phone:
        return False
    c = conn.cursor()
    c.execute("SELECT uid FROM users WHERE phone = ? AND phone != ''", (phone,))
    return c.fetchone() is not None

def create_user(email, phone, name, age, gender, smoker, chronic_conditions, temp_conditions):
    import re
    e_str = str(email).lower().strip() if email else ""
    p_str = str(phone).strip() if phone else ""

    # ── Server-side format validation ──────────────────────────────────────
    if e_str and not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', e_str):
        return False, "Invalid email format."
    if p_str and not re.match(r'^\d{10}$', p_str):
        return False, "Phone number must be exactly 10 digits."
    if not e_str and not p_str:
        return False, "Provide at least one of: email or phone number."

    conn = _get_connection()
    try:
        # ── Independent uniqueness checks ────────────────────────────────────
        if e_str and _is_email_taken(e_str, conn):
            return False, "This email address is already registered. Please sign in or use a different email."
        if p_str and _is_phone_taken(p_str, conn):
            return False, "This phone number is already registered. Please sign in or use a different phone number."

        uid = e_str if e_str else p_str

        c = conn.cursor()
        c.execute('''
            INSERT INTO users 
            (uid, email, phone, name, age, gender, smoker, chronic_conditions, temp_conditions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            uid,
            e_str,
            p_str,
            name,
            str(age),
            gender,
            smoker,
            json.dumps(chronic_conditions),
            json.dumps(temp_conditions)
        ))
        conn.commit()
    except Exception as e:
        conn.close()
        print("Create user error:", e)
        return False, "Failed to create user in database."
    finally:
        conn.close()

    return True, get_user(email, phone)


def update_health(email, phone, smoker, chronic_conditions, temp_conditions):
    user = get_user(email, phone)
    if not user:
        return False, "User not found"
        
    uid = user['uid']
    conn = _get_connection()
    c = conn.cursor()
    
    try:
        c.execute('''
            UPDATE users
            SET smoker = ?, chronic_conditions = ?, temp_conditions = ?
            WHERE uid = ?
        ''', (
            smoker,
            json.dumps(chronic_conditions),
            json.dumps(temp_conditions),
            uid
        ))
        conn.commit()
        success = True
    except Exception as e:
        success = False
        print("Update user error:", e)
    finally:
        conn.close()
        
    if success:
        return True, get_user(email, phone)
    return False, "Failed to update user in database"
