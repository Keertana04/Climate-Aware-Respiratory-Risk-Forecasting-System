import json
import sqlite3
from pathlib import Path

DB_DIR = Path(__file__).parent / "users_db"
OLD_JSON = DB_DIR / "users.json"
NEW_DB = DB_DIR / "users.db"

def migrate():
    print("Starting migration to SQLite...")
    
    if not OLD_JSON.exists():
        print(f"No {OLD_JSON} found. Nothing to migrate.")
        # We still want to initialize the DB.
        init_db()
        return

    with open(OLD_JSON, "r", encoding="utf-8") as f:
        try:
            users_data = json.load(f)
        except Exception as e:
            print("Failed to read JSON:", e)
            users_data = {}

    init_db()

    conn = sqlite3.connect(NEW_DB)
    cursor = conn.cursor()

    count = 0
    for uid, u in users_data.items():
        comp = u.get("contact") or u.get("phone") or uid
        email = u.get("email", "")
        phone = u.get("phone", "")
        
        # Best effort mapping from legacy schema
        if not email and not phone:
            if "@" in str(comp):
                email = str(comp)
            else:
                phone = str(comp)
                
        # Handle the transition from old singular "conditions" array to chronic/temp
        chronic = u.get("chronic_conditions", u.get("conditions", []))
        temp = u.get("temp_conditions", [])

        # SQLite does not have native JSON list arrays in all versions cleanly without extension, so we use stringified JSON
        chronic_json = json.dumps(chronic)
        temp_json = json.dumps(temp)

        try:
            cursor.execute('''
                INSERT OR IGNORE INTO users 
                (uid, email, phone, name, age, gender, smoker, chronic_conditions, temp_conditions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uid),
                str(email).lower().strip() if email else "",
                str(phone).strip() if phone else "",
                u.get("name", ""),
                str(u.get("age", "")),
                u.get("gender", ""),
                str(u.get("smoker", "No")),
                chronic_json,
                temp_json
            ))
            count += 1
        except Exception as e:
            print(f"Error inserting user {uid}:", e)

    conn.commit()
    conn.close()
    
    print(f"Migration complete! Successfully ported {count} profiles to SQLite users.db.")
    print("You can safely delete users.json if the app works perfectly.")

def init_db():
    conn = sqlite3.connect(NEW_DB)
    c = conn.cursor()
    c.execute('''
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
    conn.close()

if __name__ == "__main__":
    migrate()
