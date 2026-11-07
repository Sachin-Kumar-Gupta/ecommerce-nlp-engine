import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from transformers import pipeline
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# NLTK VADER Lexicon download (for Tab 2 fast routing)
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# --- PAGE CONFIG ---
st.set_page_config(page_title="E-Commerce NLP Engine", page_icon="📈", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("NLP Solutions")
app_mode = st.sidebar.radio("Select Module:", [
    "1. Deep Learning Sentiment Analyzer", 
    "2. Support Ticket Auto-Router", 
    "3. Competitor Weakness Scanner"
])

# =========================================================
# TAB 1: DEEP LEARNING SENTIMENT ANALYZER (WITH VISUALS)
# =========================================================
if app_mode == "1. Deep Learning Sentiment Analyzer":
    st.title("🧠 Deep Learning Sentiment & Keyword Analyzer")
    st.write("Upload raw e-commerce reviews. Our Context-Aware AI will analyze sentiments and extract key pain-points visually.")

    uploaded_file = st.file_uploader("Upload Reviews (CSV format)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        # --- DYNAMIC MAPPING FEATURE ---
        st.markdown("### ⚙️ Map Your Data Columns")
        all_columns = df.columns.tolist()
        
        col_map1, col_map2 = st.columns(2)
        with col_map1:
            review_col = st.selectbox("Select Review Text Column:", all_columns)
        with col_map2:
            rating_col = st.selectbox("Select Star Rating Column (Optional):", ["No Rating Column"] + all_columns)
            
        if st.button("Run Deep Learning AI 🚀"):
            # Clean empty rows
            df = df.dropna(subset=[review_col])
            
            # Limit data for fast demo execution (Remove this limit for actual client work)
            if len(df) > 500:
                st.warning("Large dataset detected. Analyzing the first 500 rows for faster preview...")
                df = df.head(500)
            
            with st.spinner("🧠 AI is reading and understanding the context of the reviews... Please wait."):
                
                # HUGGING FACE AI PIPELINE
                @st.cache_resource
                def load_ai_model():
                    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
                
                sentiment_pipeline = load_ai_model()
                
                def get_hf_sentiment(text):
                    try:
                        short_text = str(text)[:500] 
                        result = sentiment_pipeline(short_text)[0]
                        label = "Positive" if result['label'] == 'POSITIVE' else "Negative"
                        score = result['score'] if label == "Positive" else -result['score']
                        return pd.Series([label, score])
                    except:
                        return pd.Series(["Neutral", 0.0])

                df[['Sentiment_Label', 'AI_Confidence_Score']] = df[review_col].apply(get_hf_sentiment)

            # --- DASHBOARD VISUALIZATION ---
            st.markdown("---")
            st.markdown("### 📊 AI Executive Summary")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Reviews Analyzed", len(df))
            positive_pct = round((len(df[df['Sentiment_Label'] == 'Positive']) / len(df)) * 100, 1)
            col2.metric("Positive Brand Sentiment", f"{positive_pct}%")
            avg_score = round(df['AI_Confidence_Score'].mean(), 2)
            col3.metric("AI Confidence Score", avg_score)

            st.markdown("---")
            
            # --- VISUALS: PIE CHART & WORDCLOUDS ---
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### Sentiment Distribution")
                fig_pie = px.pie(
                    df, names='Sentiment_Label', color='Sentiment_Label',
                    color_discrete_map={"Positive": "#00CC96", "Negative": "#EF553B", "Neutral": "#636EFA"},
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_chart2:
                st.markdown("#### What Customers Are Saying (Keywords)")
                
                # WordCloud Generation
                positive_text = " ".join(review for review in df[df['Sentiment_Label'] == 'Positive'][review_col].astype(str))
                negative_text = " ".join(review for review in df[df['Sentiment_Label'] == 'Negative'][review_col].astype(str))
                
                stopwords = set(STOPWORDS)
                stopwords.update(["br", "href", "dress", "top", "shirt", "size", "fit", "wear", "will", "one", "really"])
                
                tab_pos, tab_neg = st.tabs(["🟢 Positive Drivers", "🔴 Negative Pain-Points"])
                
                with tab_pos:
                    if len(positive_text) > 0:
                        wordcloud_pos = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords, colormap='Greens').generate(positive_text)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud_pos, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                    else:
                        st.info("Not enough data.")
                        
                with tab_neg:
                    if len(negative_text) > 0:
                        wordcloud_neg = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords, colormap='Reds').generate(negative_text)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud_neg, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                    else:
                        st.info("Not enough data.")

            # --- DATA TABLE & EXPORT ---
            st.markdown("---")
            st.markdown("#### Detailed AI Output")
            display_cols = [review_col, 'Sentiment_Label', 'AI_Confidence_Score']
            if rating_col != "No Rating Column":
                display_cols.insert(1, rating_col)
                
            st.dataframe(df[display_cols], use_container_width=True, height=250)
            
            st.markdown("### 📥 Export Client Deliverable")
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_file = convert_df_to_csv(df)
            st.download_button(
                label="Download Analyzed Data (CSV)",
                data=csv_file,
                file_name="DeepLearning_NLP_Results.csv",
                mime="text/csv"
            )

    else:
        st.info("No file uploaded. Please upload a CSV to begin.")

# =========================================================
# TAB 2 & 3: Su[[prt Ticket and Competitor Scanner
# =========================================================
elif app_mode == "2. Support Ticket Auto-Router":
    st.title("📩 Support Ticket Auto-Router")
    st.write("Instantly classify incoming customer emails to route them to the correct department and flag urgent issues.")
    # (Aapka Tab 2 ka purana code yahan aayega jo maine pehle diya tha)

elif app_mode == "3. Competitor Weakness Scanner":
    st.title("🔍 Competitor Weakness Scanner")
    # (Aapka Tab 3 ka purana code yahan aayega)
