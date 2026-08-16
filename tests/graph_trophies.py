import csv
import matplotlib.pyplot as plt
import os
import sys

# Set paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Assuming the script is in /tests/ and data is in /data/ (one level up)
DATA_FILE = os.path.join(SCRIPT_DIR, '..', 'data', 'export_battles.csv')
OUTPUT_IMAGE = os.path.join(SCRIPT_DIR, 'player_trophy_distribution.png')

def generate_trophy_graph():
    trophies = []
    
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at {DATA_FILE}")
        return

    print(f"Reading data from {DATA_FILE}...")
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                try:
                    # Collect both player and opponent trophies to get full distribution
                    if 'player_trophies' in row and row['player_trophies']:
                        trophies.append(int(row['player_trophies']))
                    if 'opponent_trophies' in row and row['opponent_trophies']:
                        trophies.append(int(row['opponent_trophies']))
                except ValueError:
                    continue
                
                if (i + 1) % 50000 == 0:
                    print(f"Processed {i + 1} rows...")
                    
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if not trophies:
        print("No trophy data found.")
        return

    print(f"Generating histogram for {len(trophies)} data points...")
    
    plt.figure(figsize=(12, 6))
    plt.hist(trophies, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    
    plt.title('Player Trophy Distribution', fontsize=16)
    plt.xlabel('Trophies', fontsize=12)
    plt.ylabel('Frequency (Number of Players)', fontsize=12)
    plt.grid(axis='y', alpha=0.5)
    
    # Add mean and median lines
    mean_trophies = sum(trophies) / len(trophies)
    sorted_trophies = sorted(trophies)
    median_trophies = sorted_trophies[len(trophies) // 2]
    
    plt.axvline(mean_trophies, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_trophies:.0f}')
    plt.axvline(median_trophies, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_trophies:.0f}')
    plt.legend()
    
    print(f"Saving graph to {OUTPUT_IMAGE}...")
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print("Done!")

if __name__ == "__main__":
    generate_trophy_graph()
