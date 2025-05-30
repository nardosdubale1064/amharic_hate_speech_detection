# 🗣️ Amharic Social Media Hate Speech Analyzer

A web-based NLP tool for detecting **hate speech**, **offensive**, and **normal** content in **Amharic** text collected from public Telegram channels and groups. Built for educational and research purposes, this project demonstrates practical applications of Natural Language Processing for low-resource languages.

## 🚀 Features

* **🔍 Web Interface**
  Simple and clean web form where users input Telegram URLs to view categorized analysis results.

* **🧹 Amharic Text Preprocessing**
  Custom normalization for Amharic characters, removal of URLs, mentions, hashtags, emojis, and a comprehensive stopword list.

* **✂️ Sentence-Level Tokenization**
  Breaks down long messages and comments into individual sentences for more detailed classification.

* **🧠 Machine Learning Model**
  Logistic Regression classifier using TF-IDF features, trained on [`uhhlt/amharichatespeechranlp`](https://huggingface.co/datasets/uhhlt/amharichatespeechranlp) dataset.

* **📡 Telegram Scraper (Telethon)**
  Asynchronously fetches:

  * Comments from a specific Telegram post.
  * Messages (and their comments) from channels or groups.
  * Automatically detects message type (channel/group) and supports a custom message limit (default: 1000).

* **📊 Result Summary**
  Shows:

  * Number of messages/comments scraped
  * Total sentences analyzed
  * Category breakdown: hate, offensive, and normal
  * Example sentences for each category

* **🧩 Modular Design**
  Code is organized for readability and reusability using separate modules.

---

## 📁 Project Structure

```plaintext
amharic_hate_speech_analyzer/
├── config.py                   # Configuration settings (API keys, model paths, label mapping)
├── amharic_preprocessing.py   # Amharic text cleaning, normalization, and tokenization
├── model_trainer.py           # Model training and saving script
├── telegram_scraper.py        # Telegram data fetching logic (messages + comments)
├── server.py                  # Main Flask web application
├── requirements.txt           # Python dependencies
├── models/                    # Trained model and vectorizer
│   ├── amharic_hate_speech_model.pkl
│   └── tfidf_vectorizer.pkl
├── templates/                 # Web templates
│   ├── index.html
│   └── results.html
└── README.md                  # This file
```

---

## ⚙️ Setup Instructions (Local Development)

### 1. Clone and Prepare the Project Directory

```bash
git clone https://github.com/nardosdubale1064/amharic_hate_speech_detection.git
cd amharic_hate_speech_detection
```

### 2. Create and Activate a Virtual Environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Your Telegram API Credentials

Create a `.env` file or set environment variables:

```
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_SESSION_STRING=your_session_string
```

> **Note:** You must get these credentials from [my.telegram.org](https://my.telegram.org).

### 5. Train the Model (if needed)

```bash
python model_trainer.py
```

> This will create the model and vectorizer files in the `models/` folder.

### 6. Run the Flask App

```bash
python server.py
```

Visit `http://127.0.0.1:5000/` in your browser.

---

## 🌍 Live Demo (If Available)

Try the hosted version:
👉 [https://amharic-hate-speech-detection-1f9y.onrender.com](https://amharic-hate-speech-detection-1f9y.onrender.com)

---

## 🧠 Future Improvements

* Use Amharic BERT or XLM-R for better accuracy
* Add real-time monitoring and alerting system
* Improve UI with charts and graphs
* Support other Ethiopian languages

---

## 📜 License

This project is for educational and research use only. Please do not use it to collect or analyze user data without permission.

---
