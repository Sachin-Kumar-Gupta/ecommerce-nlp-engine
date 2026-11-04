# 🚀 E-Commerce NLP & Sentiment Analysis Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ecommerce-analytix.streamlit.app/)

## 📌 The Business Problem
Direct-to-Consumer (DTC) brands and e-commerce founders receive thousands of reviews and support tickets daily. Manually reading this text to understand brand sentiment, route customer issues, or find competitor weaknesses is impossible. This results in slow response times, lost revenue, and missed marketing opportunities.

## 💡 The Solution
I engineered this **Natural Language Processing (NLP) web application** to automate text analysis for e-commerce brands. It instantly quantifies human emotion from raw text and automates operational workflows.

### ⚙️ Core Modules (Features)
1. **📊 Review Sentiment Analyzer:** Upload a CSV of raw customer reviews. The engine uses the VADER NLP lexicon to score text polarity (Positive/Negative/Neutral) and visualizes the brand health distribution instantly.
2. **📩 Support Ticket Auto-Router:** An AI-driven text classifier that reads incoming support emails, tags the category (e.g., Shipping, Refunds), determines urgency based on sentiment, and routes it to the correct department.
3. **🔍 Competitor Weakness Scanner:** A comparative dashboard that highlights the most frequent negative keywords from a competitor's reviews, providing actionable angles for ad campaigns.

## 🛠️ Tech Stack & Methodology
* **Frontend/UI:** `Streamlit` (Interactive web deployment)
* **Data Manipulation:** `Pandas` (Data cleaning and structuring)
* **Natural Language Processing (NLP):** `NLTK (VADER)` (Lexicon and rule-based sentiment analysis highly optimized for social media and e-commerce text)
* **Data Visualization:** `Plotly Express` (Dynamic, interactive charting)

## 💻 How to Run Locally
If you want to test this engine on your local machine:

1. Clone this repository:
   ```bash
   git clone [https://github.com/yourusername/ecommerce-nlp-engine.git](https://github.com/yourusername/ecommerce-nlp-engine.git)
