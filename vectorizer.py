import os
import json
import csv
import pickle
import numpy as np

# Paths
DATA_DIR = "data"
VECTORIZED_DIR = os.path.join(DATA_DIR, "vectorized")
CSV_FILE = os.path.join(DATA_DIR, "export_battles.csv")
CARD_DATA_FILE = os.path.join(DATA_DIR, "cardData.json")
OUTPUT_X = os.path.join(VECTORIZED_DIR, "xTrain.npy")
OUTPUT_Y = os.path.join(VECTORIZED_DIR, "yTrain.npy")
MAP_JSON = os.path.join(VECTORIZED_DIR, "featureIndexMap.json")
MAP_PKL = os.path.join(VECTORIZED_DIR, "featureIndexMap.pkl")
META_JSON = os.path.join(VECTORIZED_DIR, "vectorizationMetadata.json")



def main():
    if not os.path.exists(VECTORIZED_DIR):
        os.makedirs(VECTORIZED_DIR)
        
    print("Phase 1: Dynamic Feature Indexing")
    
    unique_features = set()
    max_trophies = 0
    match_count = 0
    
    # First pass: Discovery
    print(f"Scanning {CSV_FILE} for unique cards and max trophies")
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            match_count += 1
            
            # Trophies
            try:
                p_trophies = int(row['player_trophies'])
                o_trophies = int(row['opponent_trophies'])
                max_trophies = max(max_trophies, p_trophies, o_trophies)
            except ValueError:
                pass
            
            # Loadouts
            for loadout_str in [row['player_loadout'], row['opponent_loadout']]:
                try:
                    loadout = json.loads(loadout_str)
                    for card_id in loadout.keys():
                        unique_features.add(card_id)
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse loadout in row {match_count}")
                    continue

    sorted_features = sorted(list(unique_features))
    feature_index_map = {feat: idx for idx, feat in enumerate(sorted_features)}
    index_feature_map = {idx: feat for idx, feat in enumerate(sorted_features)}
    
    n_features = len(sorted_features)
    print(f"Found {n_features} unique card features.")
    print(f"Max trophies found: {max_trophies}")
    print(f"Total matches: {match_count}")
    
    # Save mappings
    with open(MAP_JSON, 'w', encoding='utf-8') as f:
        json.dump(feature_index_map, f, indent=4)
        
    with open(MAP_PKL, 'wb') as f:
        pickle.dump(feature_index_map, f)
        
    print("Mappings saved.")

    print("Phase 2: Matrix Initialization")
    # Columns: (nFeatures * 2) + 5
    # [P1 Cards] [P2 Cards] [P1 Trophies] [P2 Trophies] [P1 Avg Lvl] [P2 Avg Lvl] [Trophy Diff]
    n_cols = (n_features * 2) + 5
    print(f"Matrix dimensions: {match_count} x {n_cols}")
    
    # Use memory mapping to avoid OOM
    feature_matrix = np.lib.format.open_memmap(OUTPUT_X, mode='w+', dtype=np.float32, shape=(match_count, n_cols))
    labels = np.lib.format.open_memmap(OUTPUT_Y, mode='w+', dtype=np.int32, shape=(match_count,))
    
    print("Phase 3: Vectorization Loop")
    
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader):
            # Label
            try:
                outcome = int(row['outcome'])
                labels[i] = outcome
            except ValueError:
                # Should not happen if filtered correctly
                pass
            
            p_total_level = 0.0
            p_card_count = 0
            o_total_level = 0.0
            o_card_count = 0

            # Player Loadout
            try:
                p_loadout = json.loads(row['player_loadout'])
                for card_id, level in p_loadout.items():
                    p_total_level += float(level)
                    p_card_count += 1
                    if card_id in feature_index_map:
                        idx = feature_index_map[card_id]
                        weight = float(level) / 16.0
                        feature_matrix[i, idx] = weight
            except (json.JSONDecodeError, ValueError):
                pass

            # Opponent Loadout (Offset by n_features)
            try:
                o_loadout = json.loads(row['opponent_loadout'])
                for card_id, level in o_loadout.items():
                    o_total_level += float(level)
                    o_card_count += 1
                    if card_id in feature_index_map:
                        idx = feature_index_map[card_id]
                        # Offset
                        col_idx = idx + n_features
                        weight = float(level) / 16.0
                        feature_matrix[i, col_idx] = weight
            except (json.JSONDecodeError, ValueError):
                pass
                
            # Metadata Features
            try:
                p_trophies = float(row['player_trophies'])
                o_trophies = float(row['opponent_trophies'])
                
                # 1. P1 Trophies (Norm)
                feature_matrix[i, -5] = p_trophies / max_trophies if max_trophies > 0 else 0
                
                # 2. P2 Trophies (Norm)
                feature_matrix[i, -4] = o_trophies / max_trophies if max_trophies > 0 else 0
                
                # 3. P1 Avg Level (Norm / 16.0)
                p_avg = p_total_level / p_card_count if p_card_count > 0 else 0
                feature_matrix[i, -3] = p_avg / 16.0
                
                # 4. P2 Avg Level (Norm / 16.0)
                o_avg = o_total_level / o_card_count if o_card_count > 0 else 0
                feature_matrix[i, -2] = o_avg / 16.0
                
                # 5. Trophy Diff (Norm)
                trophy_diff = p_trophies - o_trophies
                feature_matrix[i, -1] = trophy_diff / max_trophies if max_trophies > 0 else 0
                
            except ValueError:
                pass
                
            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1} rows")
    
    # Save output
    feature_matrix.flush()
    labels.flush()
    
    metadata = {
        "n_matches": match_count,
        "n_features": n_features,
        "n_cols": n_cols,
        "max_trophies": max_trophies,
        "feature_index_map_file": MAP_JSON
    }
    
    with open(META_JSON, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nSaved xTrain.npy, yTrain.npy and metadata to {VECTORIZED_DIR}")

if __name__ == "__main__":
    main()
