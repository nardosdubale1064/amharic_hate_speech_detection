import os
import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import pandas as pd
import sys
import numpy as np

# --- ADD THIS IMPORT LINE ---
from amharic_preprocessing import preprocess_amharic_text
# --- END ADDITION ---

from config import VECTORIZER_PATH, MODEL_PATH, MODEL_DIR, LABEL_MAPPING

def train_and_save_model():
    print("--- Loading Amharic Hate Speech Dataset ---")
    try:
        dataset = load_dataset("uhhlt/amharichatespeechranlp", split='train')
    except Exception as e:
        print(f"Error loading dataset: {e}. Please check your internet connection or dataset name.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(dataset)

    # REVERSE_LABEL_MAPPING: Map human-readable names back to numerical labels
    REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}
    print(f"DEBUG (model_trainer): Reverse label mapping: {REVERSE_LABEL_MAPPING}", file=sys.stderr)

    # Map labels to human-readable names for initial display (this is fine)
    # The 'label' column in the dataset might already be integers, but if it's strings, this ensures consistency.
    # The Hugging Face dataset viewer shows 0, 1, 2 for labels, but if it's loaded as category strings, this mapping fixes it.
    # We should rely on the actual 'label' column which is numerical from the dataset.
    # However, if it got converted to strings implicitly by pandas/datasets for some reason, this corrects it.
    # Let's keep `df['label_name'] = df['label'].map(LABEL_MAPPING)` as is, it's for display.
    # The important part is `y = df['label'].map(REVERSE_LABEL_MAPPING)` or similar.

    print(f"Dataset loaded. Total samples: {len(df)}")
    # This line: `Series([], Name: count, dtype: int64)` for Label distribution:
    # This means `df['label_name'].value_counts()` is empty.
    # This might indicate that `df['label']` itself is empty or contains unexpected values.
    # Let's add a debug print for `df['label']` before mapping to `label_name`.
    print(f"DEBUG (model_trainer): Raw labels from dataset (first 5): {df['label'].head().tolist()}", file=sys.stderr)
    # And specifically check the `df['label']` column types
    print(f"DEBUG (model_trainer): Raw label column dtype: {df['label'].dtype}", file=sys.stderr)

    # If df['label'] is categorical (e.g., pandas CategoryDtype), .map() might not work as expected.
    # Let's ensure it's converted to a standard numeric type first.
    # Also, if `df['label']` contains the string labels directly (which it did in your previous output),
    # `df['label'].map(LABEL_MAPPING)` would fail, as LABEL_MAPPING expects integer keys.
    # The correct way is `df['label'].map(REVERSE_LABEL_MAPPING)` for `y`.

    # Let's temporarily change the Label distribution print to use the raw labels first,
    # before relying on df['label_name'] which might be broken if label mapping fails.
    print("Label distribution (raw labels):\n", df['label'].value_counts())


    print("\n--- Preprocessing Text Data ---")
    df['processed_text'] = [preprocess_amharic_text(text) for text in tqdm(df['text'], desc="Preprocessing")]

    df = df[df['processed_text'].str.strip() != '']
    print(f"Samples after preprocessing filter: {len(df)}")

    if len(df) == 0:
        print("No valid samples remaining after preprocessing. Cannot train model.", file=sys.stderr)
        sys.exit(1)

    X = df['processed_text']
    # CRITICAL FIX: Ensure y contains numerical labels (0, 1, 2) for model training.
    # If the `df['label']` column is coming as strings like 'normal', 'hate', 'offensive'
    # from the `datasets` library/pandas conversion, then this mapping is essential.
    # If it's already integers (0, 1, 2), then `.map(REVERSE_LABEL_MAPPING)` would error.
    # Let's make it robust: try to convert to int, or use the reverse mapping if it fails.
    
    # First, try to convert 'label' column to numeric, coercing errors
    y_raw = df['label']
    try:
        y_numeric = pd.to_numeric(y_raw)
        # Check if coercion actually resulted in NaNs, indicating non-numeric values
        if y_numeric.isnull().any():
            print("DEBUG (model_trainer): Raw labels contain non-numeric values. Falling back to string mapping.", file=sys.stderr)
            y = y_raw.map(REVERSE_LABEL_MAPPING)
            if y.isnull().any(): # If mapping also produces NaNs, something is seriously wrong
                print("ERROR (model_trainer): Labels could not be converted to numeric or mapped to reverse_label_mapping. Check dataset content.", file=sys.stderr)
                sys.exit(1)
        else:
            print("DEBUG (model_trainer): Raw labels are numeric. Using them directly.", file=sys.stderr)
            y = y_numeric
    except Exception as e:
        print(f"DEBUG (model_trainer): Error converting raw labels to numeric ({e}). Attempting string mapping.", file=sys.stderr)
        y = y_raw.map(REVERSE_LABEL_MAPPING)
        if y.isnull().any():
            print("ERROR (model_trainer): Labels could not be mapped to reverse_label_mapping. Check dataset content and mapping.", file=sys.stderr)
            sys.exit(1)
        
    print(f"DEBUG (model_trainer): Final y dtype for training: {y.dtype}", file=sys.stderr)
    print(f"DEBUG (model_trainer): Final y values (first 5) for training: {y.head().tolist() if isinstance(y, pd.Series) else y[:5]}", file=sys.stderr)
    print(f"DEBUG (model_trainer): Final y unique values for training: {np.unique(y).tolist()}", file=sys.stderr)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print("\n--- Training TF-IDF Vectorizer ---")
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, min_df=5, max_df=0.8)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    print(f"TF-IDF vectorizer fitted. Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

    print("\n--- Training Logistic Regression Model ---")
    model = LogisticRegression(max_iter=2000, random_state=42, solver='liblinear', class_weight='balanced')
    model.fit(X_train_tfidf, y_train)

    print("\n--- Evaluating Model Performance ---")
    y_pred = model.predict(X_test_tfidf) # This should now predict integers!

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[LABEL_MAPPING[i] for i in sorted(LABEL_MAPPING.keys())]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n--- Saving Model and Vectorizer ---")
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        joblib.dump(tfidf_vectorizer, VECTORIZER_PATH)
        joblib.dump(model, MODEL_PATH)
        print(f"Model and vectorizer saved to {MODEL_DIR}/")
    except Exception as e:
        print(f"Error saving model or vectorizer: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    train_and_save_model()