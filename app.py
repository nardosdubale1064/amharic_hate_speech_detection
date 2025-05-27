import joblib
import asyncio
import sys
from collections import Counter
import re
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import os 

# This library is crucial for running async code (Telethon) inside a sync Flask app.
# It patches asyncio to allow nested event loops.
import nest_asyncio
nest_asyncio.apply()

from config import VECTORIZER_PATH, MODEL_PATH, LABEL_MAPPING
from amharic_preprocessing import preprocess_amharic_text, tokenize_amharic_sentences
# Import the wrapped scraping functions and the main _run_telethon_client_task
from telegram_scraper import (
    get_telegram_comments_for_message,
    get_channel_or_group_content,
    parse_telegram_url,
    _run_telethon_client_task # <--- NEW: Import the wrapper
)

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Global variables for loaded model and vectorizer ---
# These are loaded once when the Flask app starts
vectorizer = None
model = None

def load_model_and_vectorizer():
    """Loads the pre-trained TF-IDF vectorizer and classification model."""
    global vectorizer, model
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)
        print("Model and vectorizer loaded successfully.", file=sys.stderr)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Model or vectorizer not found at '{VECTORIZER_PATH}' or '{MODEL_PATH}'. "
              "Please ensure 'python model_trainer.py' was run locally and the 'models/' directory is in your Git repo.", file=sys.stderr)
        sys.exit(1) # Exit if models are not found, as the app can't function
    except Exception as e:
        print(f"CRITICAL ERROR loading model or vectorizer: {e} (Type: {type(e)})", file=sys.stderr)
        sys.exit(1)

def classify_sentences(texts_to_analyze):
    """
    Sentence-tokenizes a list of texts, preprocesses each sentence, and classifies them.
    Returns:
        tuple: (list of predicted label strings, list of original sentences that were classified)
    """
    if not texts_to_analyze:
        return [], []

    sentences_for_classification = []
    original_sentences_passed_filter = []

    for text in texts_to_analyze:
        sentences_in_text = tokenize_amharic_sentences(text)
        for original_sent in sentences_in_text:
            processed_sent = preprocess_amharic_text(original_sent)
            if processed_sent.strip():
                sentences_for_classification.append(processed_sent)
                original_sentences_passed_filter.append(original_sent)

    if not sentences_for_classification:
        return [], []

    try:
        sentence_vectors = vectorizer.transform(sentences_for_classification)
        if sentence_vectors.shape[0] == 0 or sentence_vectors.nnz == 0:
            return [], []

        predictions = model.predict(sentence_vectors)
        
        predicted_label_strings = [LABEL_MAPPING.get(p_idx, 'unknown') for p_idx in predictions]
        
        return predicted_label_strings, original_sentences_passed_filter

    except Exception as e:
        print(f"ERROR: An exception of type {type(e)} occurred during sentence classification: {e}", file=sys.stderr)
        return [], []

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
async def analyze():
    url = request.form.get('url')
    message_limit_str = request.form.get('message_limit')
    
    if not url:
        return render_template('results.html', error="Please provide a Telegram URL.")

    if not re.match(r'https?://(?:www\.)?\S+', url):
        return render_template('results.html', error="Invalid URL format. Please enter a valid URL starting with http:// or https://")

    url_info = await parse_telegram_url(url)
    url_info['original_url'] = url

    if url_info['type'] == 'invalid':
        return render_template('results.html', error="Invalid Telegram URL format. Please enter a valid Telegram channel/group or post URL.")
    elif "facebook.com/" in url:
        return render_template('results.html', error="Facebook scraping is not supported for this project. Please use Telegram URLs.")

    analysis_results = {
        'url': url,
        'error': None,
        'summary': {},
        'samples': {'hate': [], 'offensive': [], 'normal': []}
    }

    try:
        # Call the appropriate async scraper function wrapped by _run_telethon_client_task
        if url_info['type'] == 'channel_or_group':
            message_limit = 1000 # Default
            if message_limit_str and message_limit_str.lower() != 'all':
                try:
                    limit = int(message_limit_str)
                    if limit > 0:
                        message_limit = limit
                except ValueError:
                    pass
            elif message_limit_str and message_limit_str.lower() == 'all':
                message_limit = None

            channel_content_data, messages_with_comments_count, total_comments_scraped = \
                await get_channel_or_group_content(url_info['identifier'], message_limit) # This now includes client lifecycle

            if not channel_content_data:
                analysis_results['error'] = f"No content retrieved from Telegram channel/group: {url}. This might mean the channel/group is private, inaccessible, or had no recent messages with text content."
                return render_template('results.html', results=analysis_results)

            all_texts_for_classification = []
            total_messages_scraped = len(channel_content_data)
            
            for item in channel_content_data:
                all_texts_for_classification.append(item['message_text'])
                all_texts_for_classification.extend(item['comments'])

            predicted_labels_only, classified_original_sentences = classify_sentences(all_texts_for_classification)

            if not predicted_labels_only:
                analysis_results['error'] = "No sentences were classified after preprocessing and model prediction. This might mean all texts were filtered out (e.g., all emojis, links, or stopwords) or an error occurred during classification."
                return render_template('results.html', results=analysis_results)

            label_counts = Counter(predicted_labels_only)
            total_classified_sentences = len(predicted_labels_only)

            analysis_results['summary'] = {
                'type': 'Channel/Group Analysis',
                'total_messages_scraped': total_messages_scraped,
                'messages_with_comments': messages_with_comments_count,
                'total_comments_scraped': total_comments_scraped,
                'total_sentences_classified': total_classified_sentences
            }
            ordered_labels = ['hate', 'offensive', 'normal']
            for label in ordered_labels:
                count = label_counts.get(label, 0)
                percentage = (count / total_classified_sentences) * 100 if total_classified_sentences > 0 else 0
                analysis_results['summary'][label] = {'count': count, 'percentage': f"{percentage:.2f}%"}

            samples_displayed = {label: 0 for label in ordered_labels}
            for i, original_sent in enumerate(classified_original_sentences):
                predicted_label = predicted_labels_only[i]
                if samples_displayed[predicted_label] < 3:
                    analysis_results['samples'][predicted_label].append(original_sent[:200] + '...')
                    samples_displayed[predicted_label] += 1
                if all(count >= 3 for count in samples_displayed.values()):
                    break

        elif url_info['type'] == 'message':
            comments = await get_telegram_comments_for_message(url_info['identifier'], url_info['message_id']) # This now includes client lifecycle

            if not comments:
                analysis_results['error'] = f"No comments retrieved from Telegram Post: {url}. This might mean the post has no comments, or the channel/group is private/inaccessible."
                return render_template('results.html', results=analysis_results)

            predicted_labels_only, classified_original_sentences = classify_sentences(comments)

            if not predicted_labels_only:
                analysis_results['error'] = "No sentences were classified from comments after preprocessing and model prediction. This might mean all comments were filtered out (e.g., all emojis, links, or stopwords) or an error occurred during classification."
                return render_template('results.html', results=analysis_results)

            label_counts = Counter(predicted_labels_only)
            total_classified_sentences = len(predicted_labels_only)

            analysis_results['summary'] = {
                'type': 'Post Comments Analysis',
                'total_comments_retrieved': len(comments),
                'total_sentences_classified': total_classified_sentences
            }
            ordered_labels = ['hate', 'offensive', 'normal']
            for label in ordered_labels:
                count = label_counts.get(label, 0)
                percentage = (count / total_classified_sentences) * 100 if total_classified_sentences > 0 else 0
                analysis_results['summary'][label] = {'count': count, 'percentage': f"{percentage:.2f}%"}

            samples_displayed = {label: 0 for label in ordered_labels}
            for i, original_sent in enumerate(classified_original_sentences):
                predicted_label = predicted_labels_only[i]
                if samples_displayed[predicted_label] < 3:
                    analysis_results['samples'][predicted_label].append(original_sent[:200] + '...')
                    samples_displayed[predicted_label] += 1
                if all(count >= 3 for count in samples_displayed.values()):
                    break

    except ConnectionRefusedError:
        analysis_results['error'] = "Telegram authorization failed. This usually means the API ID/HASH or Session String environment variables are incorrect, expired, or not set on the server."
    except ValueError as e:
        analysis_results['error'] = f"Telegram Entity Error: {e}. Please check the URL and your Telegram account access."
    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}", file=sys.stderr)
        analysis_results['error'] = f"An unexpected server error occurred during analysis: {e}"
    finally:
        pass 

    return render_template('results.html', results=analysis_results)

# --- Application Startup ---
load_model_and_vectorizer()

# NO global Telethon client connection at startup needed anymore
# Telethon client lifecycle is managed per request within _run_telethon_client_task

if __name__ == '__main__':
    print("Starting Flask application...", file=sys.stderr)
    app.run(debug=False, host='0.0.0.0', port=os.environ.get('PORT', 5000))