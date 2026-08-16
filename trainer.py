import os
import json
import random
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# Paths
DATA_DIR = "data"
VECTORIZED_DIR = os.path.join(DATA_DIR, "vectorized")
INPUT_X = os.path.join(VECTORIZED_DIR, "xTrain.npy")
INPUT_Y = os.path.join(VECTORIZED_DIR, "yTrain.npy")
MODEL_FILE = "xgboost_model.json"
MAP_JSON = os.path.join(VECTORIZED_DIR, "featureIndexMap.json")
META_JSON = os.path.join(VECTORIZED_DIR, "vectorizationMetadata.json")
CARD_DATA_FILE = os.path.join(DATA_DIR, "cardData.json")

def load_card_names():
    if os.path.exists(CARD_DATA_FILE):
        with open(CARD_DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("id_to_name", {})
    return {}

def run_validator(X, y, model=None, X_test=None, y_test=None):
    print("\nExtended Data & Model Validation")
    
    if not os.path.exists(MAP_JSON) or not os.path.exists(META_JSON):
        print("Warning: Metadata files not found. Skipping detailed data validation.")
        return

    with open(MAP_JSON, 'r') as f:
        feature_map = json.load(f)
    
    with open(META_JSON, 'r') as f:
        metadata = json.load(f)

    has_n_features = "n_features" in metadata
    has_max_trophies = "max_trophies" in metadata

    if has_n_features:
        n_features = metadata.get("n_features")
    else:
        print("Warning: 'n_features' not found in metadata. Skipping sparsity/feature checks.")

    if has_max_trophies:
        max_trophies = metadata.get("max_trophies")
    else:
        print("Warning: 'max_trophies' not found in metadata. Skipping trophy-based displays.")
    
    # Sparsity Check
    if has_n_features:
        print("\n[Sparsity Check]")
        p1_slice = X[:, :n_features]
        p2_slice = X[:, n_features:2*n_features]
        
        avg_nz_p1 = np.count_nonzero(p1_slice, axis=1).mean()
        avg_nz_p2 = np.count_nonzero(p2_slice, axis=1).mean()
        
        print(f"Avg non-zero entries (Player 1): {avg_nz_p1:.2f} (Expected ~8-9)")
        print(f"Avg non-zero entries (Player 2): {avg_nz_p2:.2f} (Expected ~8-9)")
        
        if 7.0 <= avg_nz_p1 <= 10.0 and 7.0 <= avg_nz_p2 <= 10.0:
            print("PASS: Sparsity looks reasonable.")
        else:
            print("WARNING: Sparsity is outside expected range (7-10).")
    else:
        print("Skipping sparsity check because 'n_features' is missing in metadata.")

    # Range Check
    print("\n[Range Check]")
    min_val = X.min()
    max_val = X.max()
    print(f"Min Value: {min_val}")
    print(f"Max Value: {max_val}")
    
    if min_val >= -0.1 and max_val <= 2.0:
        print("PASS: Values are within positive range ~0-2.")
    else:
        print("WARNING: Values are outside expected range.")
        
    # Feature Integrity
    if has_n_features:
        print("\n[Feature Integrity]")
        base_count = 0
        evo_count = 0
        hero_count = 0
        tower_count = 0
        
        for card_id in feature_map.keys():
            if card_id.startswith("159"):
                tower_count += 1
            elif "_EVO" in card_id:
                evo_count += 1
            elif "_HERO" in card_id:
                hero_count += 1
            else:
                base_count += 1
                
        print(f"Feature Map counts:")
        print(f"  Tower Troops: {tower_count}")
        print(f"  Heroes: {hero_count}")
        print(f"  Evolutions: {evo_count}")
        print(f"  Base Cards: {base_count}")
        print(f"  Total: {tower_count + hero_count + evo_count + base_count} (Should be {n_features})")
    else:
        print("Skipping feature integrity checks because 'n_features' is missing in metadata.")


def main():
    print("Loading data...")
    if not os.path.exists(INPUT_X) or not os.path.exists(INPUT_Y):
        print("Error: Vectorized data not found. Please run vectorizer.py first.")
        return

    # Use mmap_mode='r' to allow data to be swapped in/out of RAM as needed
    X = np.load(INPUT_X, mmap_mode='r')
    y = np.load(INPUT_Y, mmap_mode='r')

    print(f"Loaded X: {X.shape}, y: {y.shape}")

    # D. Training Execution
    # Shuffle indices before splitting
    np.random.seed(42)
    indices = np.random.permutation(X.shape[0])
    split_idx = int(X.shape[0] * 0.8)
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Advanced indexing into memmap loads the subset into memory.
    # Since dataset is ~1.4GB, this should fit comfortably in RAM.
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # C. Model Architecture
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        objective='binary:logistic',
        tree_method='hist'  # Memory and speed optimization
    )

    print("Training model...")
    # "It evaluates the model against the 20% test set after every round."
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )

    print("Saving model...")
    model.save_model(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

    # B. Model Verification (trainer.py)
    print("\n--- Model Verification ---")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f} (Threshold > 0.60: {'PASS' if acc > 0.60 else 'FAIL'})")

    # ROC AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {roc_auc:.4f} (Threshold > 0.65: {'PASS' if roc_auc > 0.65 else 'FAIL'})")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix:")
    print(f"  True Positives (Predicted Win, Actual Win): {tp}")
    print(f"  True Negatives (Predicted Loss, Actual Loss): {tn}")
    print(f"  False Positives (Predicted Win, Actual Loss): {fp}")
    print(f"  False Negatives (Predicted Loss, Actual Win): {fn}")

    # Run Extended Validation
    run_validator(X, y, model, X_test, y_test)

if __name__ == "__main__":
    main()
