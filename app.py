import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# NLTK VADER Lexicon download (runs only once)
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# --- PAGE CONFIG ---
st.set_page_config(page_title="E-Commerce NLP Engine", page_icon="📈", layout="wide")

# --- DUMMY DATA GENERATOR ---
def get_dummy_data():
    data = {
        "Date": ["2023-10-01", "2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-06"],
        "Star_Rating": [5, 2, 4, 1, 5, 3],
        "Review_Text": [
            "Absolutely love this product! The quality is amazing and shipping was fast.",
            "Shipping was delayed by 5 days. Very disappointed with the service.",
            "Good value for money, but the packaging could be slightly better.",
            "Terrible customer service and broken item inside the box.",
            "Perfect fit! Will definitely buy again from this brand.",
            "It's okay, neither good nor bad. Just average."
        ]
    }
    return pd.DataFrame(data)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("NLP Solutions")
app_mode = st.sidebar.radio("Select Module:", [
    "1. Review Sentiment Analyzer", 
    "2. Support Ticket Auto-Router", 
    "3. Competitor Weakness Scanner"
])

# =========================================================
# TAB 1: REVIEW SENTIMENT ANALYZER
# =========================================================
if app_mode == "1. Review Sentiment Analyzer":
    st.title("📊 Customer Review Sentiment Analyzer")
    st.write("Upload raw e-commerce reviews to instantly quantify brand sentiment using Natural Language Processing.")

    # File Uploader
    uploaded_file = st.file_uploader("Upload Reviews (CSV format)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        # --- NEW DYNAMIC MAPPING FEATURE ---
        st.markdown("### ⚙️ Map Your Data Columns")
        all_columns = df.columns.tolist()
        
        col_map1, col_map2 = st.columns(2)
        with col_map1:
            review_col = st.selectbox("Select Review Text Column:", all_columns)
        with col_map2:
            rating_col = st.selectbox("Select Star Rating Column (Optional):", ["No Rating Column"] + all_columns)
            
        if st.button("Run NLP Analysis 🚀"):
            # NLP Processing (VADER Sentiment)
            sia = SentimentIntensityAnalyzer()
            
            # Apply logic on the user-selected column
            df['Sentiment_Scores'] = df[review_col].astype(str).apply(lambda x: sia.polarity_scores(x))
            df['Compound_Score'] = df['Sentiment_Scores'].apply(lambda x: x['compound'])
            
            # Categorize Sentiment
            def get_sentiment_label(score):
                if score >= 0.05: return "Positive"
                elif score <= -0.05: return "Negative"
                else: return "Neutral"
                
            df['Sentiment_Label'] = df['Compound_Score'].apply(get_sentiment_label)

            # --- DASHBOARD VISUALIZATION ---
            st.markdown("---")
            st.markdown("### Executive Summary")
            
            # Top Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Reviews Analyzed", len(df))
            
            positive_pct = round((len(df[df['Sentiment_Label'] == 'Positive']) / len(df)) * 100, 1)
            col2.metric("Positive Sentiment %", f"{positive_pct}%")
            
            avg_score = round(df['Compound_Score'].mean(), 2)
            col3.metric("Average NLP Score", avg_score)

            st.markdown("---")
            
            # Charts Section
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### Sentiment Distribution")
                fig_pie = px.pie(
                    df, 
                    names='Sentiment_Label', 
                    color='Sentiment_Label',
                    color_discrete_map={"Positive": "#00CC96", "Negative": "#EF553B", "Neutral": "#636EFA"},
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_chart2:
                st.markdown("#### Detailed NLP Output")
                
                # Smart Table Display based on User Selection
                display_cols = [review_col, 'Sentiment_Label', 'Compound_Score']
                if rating_col != "No Rating Column":
                    display_cols.insert(1, rating_col)
                    
                st.dataframe(
                    df[display_cols],
                    use_container_width=True,
                    height=350
                )

    else:
        st.info("No file uploaded. Please upload a CSV to begin.")

# =========================================================
# TAB 2: SUPPORT TICKET AUTO-ROUTER
# =========================================================
elif app_mode == "2. Support Ticket Auto-Router":
    st.title("📩 Support Ticket Auto-Router")
    st.write("Instantly classify incoming customer emails to route them to the correct department and flag urgent issues.")

    # Dropdown with sample tickets for easy demonstration
    st.markdown("**Try a sample ticket or paste your own:**")
    sample_ticket = st.selectbox("Select a sample text:", [
        "Custom Input (Type below)",
        "Where is my order? It's been 10 days and tracking hasn't updated. I need this for a birthday!",
        "The shirt I received has a hole in the sleeve. I want a full refund immediately.",
        "Hi, do you restock the blue sneakers in size 10? Thanks!"
    ])

    # Text Area for Ticket Input
    if sample_ticket == "Custom Input (Type below)":
        ticket_text = st.text_area("Paste customer email/ticket here:", height=150)
    else:
        ticket_text = st.text_area("Paste customer email/ticket here:", value=sample_ticket, height=150)

    # Analyze Button
    if st.button("Route Ticket 🚀"):
        if ticket_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            # 1. AI Categorization Logic (Keyword based routing)
            text_lower = ticket_text.lower()
            
            category = "General Inquiry"
            department = "Tier 1 Support"
            
            if any(word in text_lower for word in ['where', 'tracking', 'shipping', 'late', 'delayed', 'delivery']):
                category = "Shipping & Logistics"
                department = "Logistics Team"
            elif any(word in text_lower for word in ['refund', 'return', 'money back', 'cancel']):
                category = "Returns & Refunds"
                department = "Billing & Returns"
            elif any(word in text_lower for word in ['broken', 'hole', 'defect', 'damaged', 'not working']):
                category = "Product Defect"
                department = "Quality Assurance (QA)"

            # 2. Priority & Sentiment Logic using VADER
            sia = SentimentIntensityAnalyzer()
            sentiment_score = sia.polarity_scores(ticket_text)['compound']
            
            priority = "Normal 🟢"
            # If highly negative sentiment OR it's a refund/defect issue, flag as High Priority
            if sentiment_score <= -0.4 or category in ["Returns & Refunds", "Product Defect"]:
                priority = "High / Urgent 🔴"
            elif sentiment_score >= 0.5:
                priority = "Low (Positive) 🔵"

            # 3. Output UI (Metrics layout)
            st.markdown("### 🤖 AI Routing Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"**Category Tag:**\n\n{category}")
            with col2:
                # Dynamic color for priority
                if "High" in priority:
                    st.error(f"**Priority Level:**\n\n{priority}")
                else:
                    st.success(f"**Priority Level:**\n\n{priority}")
            with col3:
                st.warning(f"**Routed To:**\n\n{department}")
                
            st.markdown("---")
            
            # 4. Suggested Auto-Reply Draft
            st.markdown("#### Suggested Auto-Reply Draft")
            if category == "Shipping & Logistics":
                st.write("> *Hi there, we sincerely apologize for the delay in your shipment. I have escalated this to our Logistics Team to check the exact status with our carrier right away...*")
            elif category == "Returns & Refunds":
                st.write("> *Hi there, I am sorry to hear you'd like a refund. I have routed your ticket to the Billing Team who will process your request within 24 hours...*")
            elif category == "Product Defect":
                st.write("> *Hi there, I am so sorry about the damaged item. We have notified our QA team. A replacement will be dispatched immediately...*")
            else:
                st.write("> *Hi there, thank you for reaching out! Our Tier 1 Support team is reviewing your message and will get back to you shortly...*")

# =========================================================
# TAB 3: COMPETITOR WEAKNESS SCANNER
# =========================================================
elif app_mode == "3. Competitor Weakness Scanner":
    st.title("🔍 Competitor Weakness Scanner")
    st.write("Compare negative reviews between your brand and competitors to find marketing advantages.")

    col_brand1, col_brand2 = st.columns(2)
    
    with col_brand1:
        st.markdown("### Your Brand (Sample Data)")
        your_weaknesses = {
            "Weakness Area": ["Pricing", "Battery Life", "App Glitches"],
            "Negative Mentions": [45, 12, 5]
        }
        df_yours = pd.DataFrame(your_weaknesses)
        st.dataframe(df_yours, use_container_width=True, hide_index=True)
        
    with col_brand2:
        st.markdown("### Top Competitor (Sample Data)")
        comp_weaknesses = {
            "Weakness Area": ["Customer Service", "Shipping Delays", "Build Quality"],
            "Negative Mentions": [120, 85, 60]
        }
        df_comp = pd.DataFrame(comp_weaknesses)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 💡 Recommended Marketing Angle")
    st.success("Target the competitor's biggest flaw! In your next ad campaign, emphasize your **fast shipping** and **superior 24/7 customer service**.")
