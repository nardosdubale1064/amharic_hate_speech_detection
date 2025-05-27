import joblib
import asyncio
import sys
from collections import Counter
import re
import numpy as np

from config import VECTORIZER_PATH, MODEL_PATH, LABEL_MAPPING
from amharic_preprocessing import preprocess_amharic_text, tokenize_amharic_sentences
from telegram_scraper import get_telegram_comments_for_message, get_channel_or_group_content, parse_telegram_url

# --- Global variables to store loaded model and vectorizer ---
vectorizer = None
model = None

def load_model_and_vectorizer():
    """Loads the pre-trained TF-IDF vectorizer and classification model."""
    global vectorizer, model
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)
        print("Model and vectorizer loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model or vectorizer not found at '{VECTORIZER_PATH}' or '{MODEL_PATH}'.", file=sys.stderr)
        print("Please run 'python model_trainer.py' first to train and save the model.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model or vectorizer: {e} (Type: {type(e)})", file=sys.stderr)
        sys.exit(1)

def classify_sentences(texts_to_analyze):
    """
    Sentence-tokenizes a list of texts, preprocesses each sentence, and classifies them.
    Returns:
        list: A list of tuples (original_sentence_text, predicted_label_string) for all classified sentences.
    """
    if not texts_to_analyze:
        return []

    sentences_for_classification = [] # List of preprocessed sentences
    original_sentences = []          # List of original sentences corresponding to preprocessed ones

    for text in texts_to_analyze:
        sentences_in_text = tokenize_amharic_sentences(text)
        for original_sent in sentences_in_text:
            processed_sent = preprocess_amharic_text(original_sent)
            if processed_sent.strip():
                sentences_for_classification.append(processed_sent)
                original_sentences.append(original_sent)
            # else:
                # print(f"DEBUG: Filtered out sentence: '{original_sent[:50]}...' (became empty after preprocessing)", file=sys.stderr)


    if not sentences_for_classification:
        return []

    try:
        sentence_vectors = vectorizer.transform(sentences_for_classification)
        if sentence_vectors.shape[0] == 0 or sentence_vectors.nnz == 0:
            return []

        predictions = model.predict(sentence_vectors)
        
        classified_results = []
        for i, pred_label_index in enumerate(predictions):
            predicted_label_string = LABEL_MAPPING.get(pred_label_index, 'unknown')
            classified_results.append((original_sentences[i], predicted_label_string))
        
        return classified_results

    except Exception as e:
        print(f"ERROR: An exception of type {type(e)} occurred during sentence classification: {e}", file=sys.stderr)
        return []


async def analyze_channel_or_group_content(url_info, message_limit):
    """Analyzes messages and comments for an entire Telegram channel or group."""
    identifier = url_info['identifier']
    print(f"\n--- Starting Analysis for Telegram Channel/Group: {url_info['original_url']} ---")
    print(f"Attempting to scrape up to {message_limit if message_limit else 'all'} recent messages and their comments.")

    channel_content_data, messages_with_comments_count, total_comments_scraped = \
        await get_channel_or_group_content(identifier, message_limit)

    if not channel_content_data:
        print(f"No content retrieved from Telegram channel/group: {url_info['original_url']}. Analysis aborted.", file=sys.stderr)
        return

    all_texts_for_classification = []
    total_messages_scraped = len(channel_content_data)
    
    for item in channel_content_data:
        all_texts_for_classification.append(item['message_text'])
        all_texts_for_classification.extend(item['comments']) # Add all comments

    print(f"Successfully retrieved {total_messages_scraped} messages and {total_comments_scraped} comments from the channel/group.")
    
    # Classify all the gathered texts sentence-by-sentence
    classified_sentences_data = classify_sentences(all_texts_for_classification)

    if not classified_sentences_data:
        print("No sentences were classified after preprocessing and model prediction. Analysis aborted.", file=sys.stderr)
        return

    # Aggregate results
    predicted_labels_only = [label for _, label in classified_sentences_data]
    label_counts = Counter(predicted_labels_only)
    total_classified_sentences = len(classified_sentences_data)

    print(f"\n--- Overall Sentiment Report for Channel/Group: {url_info['original_url']} ---")
    print("-" * 60)
    print(f"Total Messages Scraped      : {total_messages_scraped}")
    print(f"Messages with Comments      : {messages_with_comments_count}")
    print(f"Total Comments Scraped      : {total_comments_scraped}")
    print(f"Total Sentences Classified  : {total_classified_sentences}")
    print("-" * 60)

    ordered_labels = ['hate', 'offensive', 'normal']
    for label in ordered_labels:
        count = label_counts.get(label, 0)
        percentage = (count / total_classified_sentences) * 100 if total_classified_sentences > 0 else 0
        print(f"{label.capitalize().ljust(12)}: {count:5} ({percentage:6.2f}%)")
    print("-" * 60)

    # Display sample sentences
    print("\n--- Sample Classified Sentences (First 3 per category from Channel/Group) ---")
    samples_displayed = {label: 0 for label in ordered_labels}
    for original_sent, predicted_label in classified_sentences_data:
        if samples_displayed[predicted_label] < 3:
            print(f"[{predicted_label.upper()}] {original_sent[:150]}...") # Show first 150 chars of original sentence
            samples_displayed[predicted_label] += 1
        
        if all(count >= 3 for count in samples_displayed.values()):
            break
    print("-" * 60)


async def analyze_message_content(url_info):
    """Analyzes comments for a single Telegram message."""
    channel_identifier = url_info['identifier']
    message_id = url_info['message_id']
    print(f"\n--- Starting Analysis for Telegram Post: {url_info['original_url']} ---")

    comments = await get_telegram_comments_for_message(channel_identifier, message_id)

    if not comments:
        print(f"No comments retrieved from Telegram Post: {url_info['original_url']}. Analysis aborted.", file=sys.stderr)
        return

    print(f"Successfully retrieved {len(comments)} raw comments from Telegram.")

    # Classify the gathered comments sentence-by-sentence
    classified_sentences_data = classify_sentences(comments)

    if not classified_sentences_data:
        print("No sentences were classified after preprocessing and model prediction. Analysis aborted.", file=sys.stderr)
        return

    # Aggregate results
    predicted_labels_only = [label for _, label in classified_sentences_data]
    label_counts = Counter(predicted_labels_only)
    total_classified_sentences = len(classified_sentences_data)


    print(f"\n--- Sentiment Analysis Report for Telegram Post ---")
    print(f"URL: {url_info['original_url']}")
    print(f"Total Comments Retrieved: {len(comments)}")
    print(f"Total Sentences Classified (from comments): {total_classified_sentences}")
    print("-" * 50)

    ordered_labels = ['hate', 'offensive', 'normal']
    for label in ordered_labels:
        count = label_counts.get(label, 0)
        percentage = (count / total_classified_sentences) * 100 if total_classified_sentences > 0 else 0
        print(f"{label.capitalize().ljust(12)}: {count:5} ({percentage:6.2f}%)")
    print("-" * 50)

    # Display sample sentences
    print("\n--- Sample Classified Sentences (First 3 per category from Comments) ---")
    samples_displayed = {label: 0 for label in ordered_labels}
    for original_sent, predicted_label in classified_sentences_data:
        if samples_displayed[predicted_label] < 3:
            print(f"[{predicted_label.upper()}] {original_sent[:150]}...")
            samples_displayed[predicted_label] += 1
        if all(count >= 3 for count in samples_displayed.values()):
            break
    print("-" * 50)


async def main():
    """Main entry point for the application."""
    load_model_and_vectorizer()

    print("\n--- Amharic Social Media Hate Speech Analyzer ---")
    print("Enter a Telegram URL (channel or group or specific post) to analyze its content.")
    print("Type 'exit' to quit the application.")
    print("Note: Facebook scraping is not supported for this project.")

    while True:
        url = input("\nEnter URL (e.g., https://t.me/channel_username or https://t.me/channel_username/message_id): ")
        if url.lower() == 'exit':
            break
        if not url.strip():
            print("URL cannot be empty. Please enter a valid URL or 'exit'.", file=sys.stderr)
            continue

        # Basic URL format validation
        if not re.match(r'https?://(?:www\.)?\S+', url):
            print("Invalid URL format. Please enter a valid URL starting with http:// or https://", file=sys.sys.stderr)
            continue

        url_info = await parse_telegram_url(url)
        url_info['original_url'] = url # Store original URL for reporting

        if url_info['type'] == 'invalid':
            print("Invalid Telegram URL format. Please enter a valid Telegram channel/group or post URL.", file=sys.stderr)
            continue
        elif "facebook.com/" in url: # Redundant check but good for explicit user feedback
            print("\n--- IMPORTANT NOTE ON FACEBOOK ---", file=sys.stderr)
            print("Direct scraping of Facebook content is not supported for this project.", file=sys.stderr)
            print("Please focus on Telegram URLs.", file=sys.stderr)
            continue
        
        try:
            if url_info['type'] == 'channel_or_group':
                try:
                    message_limit_str = input(f"Enter number of recent channel/group messages to analyze (default 1000, 'all' for no limit): ")
                    if message_limit_str.lower() == 'all':
                        message_limit = None
                    elif message_limit_str.strip() == '': # User pressed enter for default
                        message_limit = 1000
                    else:
                        message_limit = int(message_limit_str)
                        if message_limit <= 0:
                            print("Message limit must be a positive number or 'all'. Using default (1000).", file=sys.stderr)
                            message_limit = 1000
                except ValueError:
                    print("Invalid limit entered. Using default (1000).", file=sys.stderr)
                    message_limit = 1000

                await analyze_channel_or_group_content(url_info, message_limit)
            elif url_info['type'] == 'message':
                await analyze_message_content(url_info)
        except Exception as e:
            print(f"An unexpected error occurred during analysis: {e}", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(main())