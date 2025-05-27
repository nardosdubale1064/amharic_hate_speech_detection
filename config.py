import os

# --- Telegram API Configuration ---
TELEGRAM_API_ID = os.environ.get('TELEGRAM_API_ID')
TELEGRAM_API_HASH = os.environ.get('TELEGRAM_API_HASH')
# New: Telegram session string to avoid interactive login on server
TELEGRAM_SESSION_STRING = os.environ.get('TELEGRAM_SESSION_STRING')
TELEGRAM_SESSION_NAME = 'amharic_hate_speech_session' # Still used for local session generation/testing

# --- Model Paths ---
MODEL_DIR = 'models'
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'amharic_hate_speech_model.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)

# --- Dataset Label Mapping ---
LABEL_MAPPING = {0: 'normal', 1: 'hate', 2: 'offensive'}