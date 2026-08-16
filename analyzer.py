import sqlite3
import csv
import os

DATA_DIR = "data"
DB_FILE = os.path.join(DATA_DIR, "raw_battles.db")
OUTPUT_FILE = os.path.join(DATA_DIR, "export_battles.csv")

def analyze():
    if not os.path.exists(DB_FILE):
        print(f"Database {DB_FILE} not found.")
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT battle_time, game_mode, outcome, trophy_diff, 
               player_trophies, opponent_trophies, player_loadout, 
               opponent_loadout, crawled_at
        FROM battles
    ''')
    
    rows = cursor.fetchall()

    # Compute total winrate across all rows
    total_battles = len(rows)
    total_wins = sum(1 for r in rows if (r[2] == 1))  # outcome at index 2
    total_winrate = (total_wins / total_battles) * 100 if total_battles else 0.0
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            "id", "battle_time", "game_mode", "outcome", "trophy_diff", 
            "player_trophies", "opponent_trophies", "player_loadout", 
            "opponent_loadout", "crawled_at"
        ])
        
        for idx, row in enumerate(rows, 1):
            writer.writerow([idx] + list(row))
            
    print(f"Exported {len(rows)} battles to {OUTPUT_FILE}")
    print(f"Total winrate: {total_winrate:.4f}%")
    conn.close()

if __name__ == "__main__":
    analyze()
