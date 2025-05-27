# Amharic Social Media Hate Speech Analyzer

This project is a comprehensive Amharic Natural Language Processing (NLP) web application designed to classify **sentences** from Telegram channel/group messages and their comments into categories: "hate speech," "offensive," or "normal." It provides a quantitative and qualitative analysis of content sentiment via a web interface.

## Features

*   **Web Interface:** User-friendly web form to input Telegram URLs and view analysis results.
*   **Robust Amharic Preprocessing:** Includes custom normalization for Amharic characters, removal of URLs, mentions, hashtags, emojis, and a comprehensive stopword list.
*   **Amharic Sentence Tokenization:** Breaks down longer texts (messages, comments) into individual sentences for more granular analysis.
*   **Machine Learning Model:** Utilizes a Logistic Regression classifier with TF-IDF features, trained on the `uhhlt/amharichatespeechranlp` dataset for Amharic hate speech detection.
*   **Telegram Content Scraper:** Leverages the `Telethon` library to asynchronously fetch:
    *   Comments for a specific Telegram channel post.
    *   Messages and their associated comments for an entire Telegram channel or group. This includes handling both channels (with linked discussion groups) and groups (where comments are direct replies).
    *   Supports a customizable limit on the number of messages to fetch from a channel/group (defaulting to 1000).
*   **Comprehensive Reporting:** Provides a summary of messages/comments scraped, how many messages had comments, the total number of sentences classified, and the percentage breakdown of hate, offensive, and normal sentences, along with sample classified sentences.
*   **Modular Design:** Code is organized into separate Python files for clarity, maintainability, and reusability.

## Project Structure

amharic_hate_speech_analyzer/
├── config.py # Configuration settings (API keys, model paths, label mapping)
├── amharic_preprocessing.py # Core logic for Amharic text cleaning, normalization, and sentence tokenization
├── model_trainer.py # Script to train and save the NLP model
├── telegram_scraper.py # Asynchronous script to scrape Telegram content (messages & comments for channels/groups)
├── server.py # Main Flask web application
├── README.md # Project documentation (this file)
├── requirements.txt # List of Python dependencies
├── models/ # Directory containing trained model and vectorizer files
│ ├── amharic_hate_speech_model.pkl
│ └── tfidf_vectorizer.pkl
└── templates/ # HTML templates for the web interface
├── index.html
└── results.html
## Setup Instructions (Local Development)

### 1. Create Project Directory and Files

First, create a folder for your project and save all the `.py` files, `requirements.txt`, and the `templates/` folder (with `index.html` and `results.html` inside) within it. Also create an empty `models/` folder.

```bash
mkdir amharic_hate_speech_analyzer
cd amharic_hate_speech_analyzer
# Then create the files and folders manually as described above.
# Don't forget to create the empty 'models' directory: mkdir models
# And the 'templates' directory with its files: mkdir templates