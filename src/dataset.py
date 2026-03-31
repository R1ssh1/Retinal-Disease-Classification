import os
import cv2
import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split

def load_odir_dataset(csv_path, img_dir, img_size=224, sample_fraction=1.0):
    """
    Loads and preprocesses the ODIR-5K dataset.
    """
    labels_df = pd.read_csv(csv_path)
    if sample_fraction < 1.0:
        labels_df = labels_df.sample(frac=sample_fraction, random_state=42)
        
    images = []
    targets = []
    
    print(f"Loading images from {img_dir}...")
    for i, row in labels_df.iterrows():
        path = os.path.join(img_dir, row['filename'])
        if not os.path.exists(path):
            continue
            
        img = cv2.imread(path)
        if img is None:
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0 # Normalize globally
        
        images.append(img)
        
        # Parse targets
        if isinstance(row["target"], str):
            # Safer parsing for serialized list labels such as "[1,0,0,1,...]"
            targets.append(ast.literal_eval(row["target"]))
        else:
            targets.append(row["target"])
            
    X = np.array(images, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    
    if len(X) == 0:
        raise ValueError("No images loaded! Please check your dataset path.")
    
    print(f"Successfully loaded {len(X)} images. Shape: X={X.shape}, y={y.shape}")

    # Stratify by dominant label (argmax) so train/val class mix is similar — reduces overfitting to split luck
    strat = None
    if len(np.unique(np.argmax(y, axis=1))) > 1:
        strat = np.argmax(y, axis=1)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=strat
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    return X_train, X_test, y_train, y_test
