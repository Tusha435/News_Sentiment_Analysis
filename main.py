# main.py
import streamlit as st
import requests
from textblob import TextBlob
import os
from gtts import gTTS
from bs4 import BeautifulSoup
import time
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()
# Set page configuration
st.set_page_config(
    page_title="News Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

# API Configuration (in a real app, these would be in .env)
NEWS_API_URL = "https://newsapi.org/v2/everything"
NEWS_API_KEY = os.getenv("NEWS_API_KEY") # For Hugging Face Spaces deployment

# News Extraction
def extract_news(company_name, num_articles=10):
    params = {
        "q": company_name,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": num_articles
    }
    
    try:
        with st.spinner(f"Fetching news for {company_name}..."):
            response = requests.get(NEWS_API_URL, params=params)
            
        if response.status_code != 200:
            st.error(f"Error: {response.status_code} - {response.text}")
            return []
            
        data = response.json()
        
        if "articles" not in data or not data["articles"]:
            st.warning("No articles found in the API response.")
            return []
            
        articles = []
        for item in data.get("articles", []):
            title = item.get("title", "No title")
            summary = item.get("description", "No summary")
            content = item.get("content", summary)
            link = item.get("url", "No link")
            published_date = item.get("publishedAt", "Unknown date")
            source = item.get("source", {}).get("name", "Unknown source")
            
            # Get full content if available (through web scraping)
            full_content = get_article_content(link)
            if full_content:
                content = full_content
                
            # Perform sentiment analysis
            sentiment_result = analyze_sentiment(content)
                
            # Extract topics
            topics = extract_topics(content)
                
            articles.append({
                'title': title,
                'summary': summary,
                'content': content,
                'link': link,
                'published_date': published_date,
                'source': source,
                'sentiment': sentiment_result,
                'topics': topics
            })
            
        return articles
            
    except Exception as e:
        st.error(f"An error occurred while fetching news: {str(e)}")
        return []

def get_article_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text content
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        
        # Clean text
        content = ' '.join(content.split())
        
        return content if content else None
        
    except Exception as e:
        # Silently fail and return None, rather than breaking the flow
        return None

# Sentiment Analysis
def analyze_sentiment(text):
    if not text:
        return {
            'label': 'Neutral',
            'polarity': 0,
            'subjectivity': 0
        }
        
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    
    if polarity > 0.1:
        label = 'Positive'
    elif polarity < -0.1:
        label = 'Negative'
    else:
        label = 'Neutral'
        
    return {
        'label': label,
        'polarity': polarity,
        'subjectivity': subjectivity
    }

def extract_topics(text):
    if not text:
        return []
        
    text = text.lower()
    topics = []
    
    topic_keywords = {
        'Financial Performance': ['revenue', 'profit', 'earnings', 'financial', 'quarter', 'fiscal'],
        'Stock Market': ['stock', 'shares', 'market', 'investors', 'trading', 'price'],
        'Products & Services': ['product', 'service', 'launch', 'release', 'announced'],
        'Management': ['ceo', 'executive', 'leadership', 'management', 'board'],
        'Partnerships': ['partnership', 'collaboration', 'deal', 'agreement', 'alliance'],
        'Regulations': ['regulation', 'compliance', 'legal', 'lawsuit', 'settlement', 'regulatory'],
        'Innovation': ['innovation', 'technology', 'research', 'development', 'patent'],
        'Sustainability': ['sustainable', 'environmental', 'green', 'carbon', 'climate']
    }
    
    for topic, keywords in topic_keywords.items():
        if any(keyword in text for keyword in keywords):
            topics.append(topic)
            
    return topics

# Comparative Analysis
def perform_comparative_analysis(articles):
    if not articles:
        return {
            'sentiment_counts': {},
            'avg_polarity': 0,
            'avg_subjectivity': 0,
            'top_topics': [],
            'sentiment_by_source': {},
            'sentiment_by_topic': {}
        }
    
    # Basic sentiment counts
    sentiment_counts = Counter(article['sentiment']['label'] for article in articles)
    
    # Calculate average polarity and subjectivity
    polarities = [article['sentiment']['polarity'] for article in articles]
    subjectivities = [article['sentiment']['subjectivity'] for article in articles]
    
    avg_polarity = sum(polarities) / len(polarities) if polarities else 0
    avg_subjectivity = sum(subjectivities) / len(subjectivities) if subjectivities else 0
    
    # Collect all topics
    all_topics = []
    for article in articles:
        all_topics.extend(article['topics'])
    
    # Get top topics
    top_topics = Counter(all_topics).most_common(5)
    
    # Sentiment by news source
    sentiment_by_source = {}
    for article in articles:
        source = article['source']
        sentiment = article['sentiment']['label']
        
        if source not in sentiment_by_source:
            sentiment_by_source[source] = []
        
        sentiment_by_source[source].append(sentiment)
    
    # Process sentiment by source
    for source in sentiment_by_source:
        sentiment_by_source[source] = Counter(sentiment_by_source[source])
    
    # Sentiment by topic
    sentiment_by_topic = {}
    for article in articles:
        sentiment = article['sentiment']['label']
        
        for topic in article['topics']:
            if topic not in sentiment_by_topic:
                sentiment_by_topic[topic] = []
            
            sentiment_by_topic[topic].append(sentiment)
    
    # Process sentiment by topic
    for topic in sentiment_by_topic:
        sentiment_by_topic[topic] = Counter(sentiment_by_topic[topic])
    
    return {
        'sentiment_counts': sentiment_counts,
        'avg_polarity': avg_polarity,
        'avg_subjectivity': avg_subjectivity,
        'top_topics': top_topics,
        'sentiment_by_source': sentiment_by_source,
        'sentiment_by_topic': sentiment_by_topic
    }

def generate_sentiment_summary(analysis_results, company_name):
    sentiment_counts = analysis_results['sentiment_counts']
    total_articles = sum(sentiment_counts.values())
    
    positive = sentiment_counts.get('Positive', 0)
    negative = sentiment_counts.get('Negative', 0)
    neutral = sentiment_counts.get('Neutral', 0)
    
    # Calculate percentages
    pos_percent = (positive / total_articles * 100) if total_articles > 0 else 0
    neg_percent = (negative / total_articles * 100) if total_articles > 0 else 0
    neu_percent = (neutral / total_articles * 100) if total_articles > 0 else 0
    
    # Overall sentiment determination
    if positive > negative and positive > neutral:
        overall = "positive"
    elif negative > positive and negative > neutral:
        overall = "negative"
    else:
        overall = "neutral"
    
    # Get top topics
    top_topics = [topic for topic, count in analysis_results['top_topics']]
    topics_text = ", ".join(top_topics) if top_topics else "No prominent topics found"
    
    # Generate summary text
    summary = f"""
    Sentiment Analysis Summary for {company_name}:
    
    Based on the analysis of {total_articles} news articles, the overall sentiment toward {company_name} is {overall}.
    
    Sentiment Distribution:
    - Positive: {positive} articles ({pos_percent:.1f}%)
    - Negative: {negative} articles ({neg_percent:.1f}%)
    - Neutral: {neutral} articles ({neu_percent:.1f}%)
    
    The average sentiment polarity is {analysis_results['avg_polarity']:.2f} (on a scale from -1 to 1).
    
    Main topics discussed in relation to {company_name} are: {topics_text}.
    """
    
    return summary

from deep_translator import GoogleTranslator

def convert_to_hindi_tts(text):
    try:
        with st.spinner("Translating to Hindi and generating audio..."):
            # Translate text to Hindi
            try:
                # Limit text length if needed (Google Translator has limits)
                text_to_translate = text[:4000] if len(text) > 4000 else text
                
                # Translate to Hindi
                translator = GoogleTranslator(source='auto', target='hi')
                hindi_text = translator.translate(text_to_translate)
                
                if not hindi_text:
                    st.warning("Translation failed. Using original text.")
                    hindi_text = text
            except Exception as trans_error:
                st.warning(f"Translation error: {str(trans_error)}. Using original text.")
                hindi_text = text
            
            # Generate speech using BytesIO to avoid file system operations
            audio_bytes = io.BytesIO()
            tts = gTTS(text=hindi_text, lang='hi', slow=False)
            tts.write_to_fp(audio_bytes)
            
            # Reset pointer to the beginning
            audio_bytes.seek(0)
            
            return audio_bytes.getvalue()
            
    except Exception as e:
        st.error(f"Error in TTS generation: {str(e)}")
        return None

import plotly.graph_objects as go  
def create_sentiment_donut_chart(sentiment_counts):
    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())
    
    # Custom colors for sentiments
    colors = {
        'Positive': '#2ecc71',  # Green
        'Negative': '#e74c3c',  # Red
        'Neutral': '#3498db'    # Blue
    }
    
    # Ensure colors match labels
    color_list = [colors.get(label, '#7f8c8d') for label in labels]

    # Create the donut chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=color_list),
                hoverinfo="label+percent+value",
                textinfo="percent",
                textfont=dict(size=16, color="white"),
                hole=0.6,  # This creates the donut effect
                pull=[0.05 if label == max(sentiment_counts, key=sentiment_counts.get) else 0 for label in labels]  # Pull out the largest segment
            )
        ]
    )

    # Add annotation in the center
    total = sum(values)
    fig.update_layout(
        title="Sentiment Distribution",
        title_font_size=20,
        title_x=0.5,  # Center the title
        annotations=[dict(
            text=f'Total<br>{total}',
            x=0.5, y=0.5,
            font_size=20,
            showarrow=False
        )],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=20, r=20, t=50, b=50),
    )

    return fig

import plotly.express as px  # Add this import at the top of the file

def create_topics_bar_chart(top_topics):
    if not top_topics:
        return None

    # Unpack topics and counts
    topics, counts = zip(*top_topics)

    # Create a DataFrame for Plotly
    data = pd.DataFrame({
        "Topics": topics,
        "Article Count": counts
    })

    # Create the bar chart
    fig = px.bar(
        data,
        x="Article Count",
        y="Topics",
        orientation="h",
        color="Article Count",
        color_continuous_scale="Blues",
        title="Top Topics in News Articles",
        labels={"Article Count": "Number of Articles", "Topics": "Topics"}
    )

    # Update layout for better appearance
    fig.update_layout(
        title_font_size=20,
        title_x=0.5,  # Center the title
        xaxis_title="Number of Articles",
        yaxis_title="Topics",
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(categoryorder="total ascending"),  # Sort topics by count
    )

    return fig


# Main application
def main():
    st.title("🔍 News Sentiment Analysis with Hindi Text-to-Speech")
    
    st.markdown("""
    This application extracts news articles about a company, analyzes sentiment,
    and provides a comparative analysis with Hindi audio output.
    """)
    
    # Company selection
    st.sidebar.header("Input Options")
    
    # Option to select from popular companies or enter custom
    company_options = ["Apple", "Google", "Microsoft", "Amazon", "Tesla", "Facebook", "Netflix", "Nvidia", "Other"]
    company_selection = st.sidebar.selectbox("Select a company:", company_options)
    
    if company_selection == "Other":
        company_name = st.sidebar.text_input("Enter company name:", "")
    else:
        company_name = company_selection
    
    # Number of articles
    num_articles = st.sidebar.slider("Number of articles to analyze:", 5, 20, 10)
    
    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        show_full_content = st.checkbox("Show full article content", value=False)
        download_report = st.checkbox("Generate downloadable report", value=True)
    
    # Main content
    if company_name:
        if st.button("Analyze News"):
            # Extract news
            articles = extract_news(company_name, num_articles)
            
            if not articles:
                st.error(f"No articles found for {company_name}. Please try another company name.")
                return
            
            # Save articles in session state for reuse
            st.session_state.articles = articles
            st.session_state.company_name = company_name
            
            # Perform comparative analysis
            analysis_results = perform_comparative_analysis(articles)
            st.session_state.analysis_results = analysis_results
            
            # Generate summary
            summary = generate_sentiment_summary(analysis_results, company_name)
            st.session_state.summary = summary
            
            # Display results
            display_results(articles, analysis_results, summary, company_name, show_full_content, download_report)
            
        # If articles are already fetched and in session state
        elif 'articles' in st.session_state and st.session_state.company_name == company_name:
            display_results(
                st.session_state.articles,
                st.session_state.analysis_results,
                st.session_state.summary,
                company_name,
                show_full_content,
                download_report
            )
    else:
        st.info("Please enter a company name and click 'Analyze News'.")

def display_results(articles, analysis_results, summary, company_name, show_full_content, download_report):
    """Display analysis results in the Streamlit UI."""
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📈 Analysis", "📰 Articles", "🔊 Audio Summary"])
    
    with tab1:
        st.header(f"Sentiment Analysis for {company_name}")
        
        # Summary 
        st.subheader("Summary")
        st.write(summary)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            donut_chart = create_sentiment_donut_chart(analysis_results['sentiment_counts'])
            if donut_chart:
                st.plotly_chart(donut_chart, use_container_width=True)
        
        with col2:
            st.subheader("Top Topics")
            topics_chart = create_topics_bar_chart(analysis_results['top_topics'])
            if topics_chart:
                st.plotly_chart(topics_chart, use_container_width=True)
        
        # Additional metrics
        st.subheader("Detailed Metrics")
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("Average Sentiment Polarity", f"{analysis_results['avg_polarity']:.2f}")
        
        with metrics_col2:
            st.metric("Average Subjectivity", f"{analysis_results['avg_subjectivity']:.2f}")
        
        with metrics_col3:
            total_articles = sum(analysis_results['sentiment_counts'].values())
            st.metric("Total Articles Analyzed", total_articles)
        
        # Generate downloadable report if requested
        if download_report:
            report_data = {
                "company": company_name,
                "summary": summary,
                "total_articles": sum(analysis_results['sentiment_counts'].values()),
                "positive_count": analysis_results['sentiment_counts'].get('Positive', 0),
                "negative_count": analysis_results['sentiment_counts'].get('Negative', 0),
                "neutral_count": analysis_results['sentiment_counts'].get('Neutral', 0),
                "avg_polarity": analysis_results['avg_polarity'],
                "avg_subjectivity": analysis_results['avg_subjectivity'],
                "top_topics": [topic for topic, _ in analysis_results['top_topics']]
            }
            
            # Convert to CSV
            report_df = pd.DataFrame([report_data])
            csv = report_df.to_csv(index=False)
            
            # Provide download button
            st.download_button(
                label="Download Analysis Report (CSV)",
                data=csv,
                file_name=f"{company_name}_sentiment_analysis.csv",
                mime="text/csv"
            )
    
    with tab2:
        st.header(f"News Articles for {company_name}")
        
        for i, article in enumerate(articles, 1):
            with st.expander(f"{i}. {article['title']} - {article['source']}"):
                st.markdown(f"**Source:** {article['source']}")
                st.markdown(f"**Published:** {article['published_date']}")
                st.markdown(f"**Summary:** {article['summary']}")
                
                if show_full_content and article['content']:
                    st.markdown("**Full Content:**")
                    st.markdown(article['content'][:1000] + "..." if len(article['content']) > 1000 else article['content'])
                
                st.markdown(f"**Sentiment:** {article['sentiment']['label']} (Polarity: {article['sentiment']['polarity']:.2f})")
                
                if article['topics']:
                    st.markdown(f"**Topics:** {', '.join(article['topics'])}")
                
                st.markdown(f"[Read full article]({article['link']})")
    
    with tab3:
        st.header("Hindi Audio Summary")
        st.markdown("Listen to the sentiment analysis summary in Hindi:")
        
        # Generate Hindi TTS
        audio_bytes = convert_to_hindi_tts(summary)
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
            
            # Provide download button for audio
            st.download_button(
                label="Download Hindi Audio",
                data=audio_bytes,
                file_name=f"{company_name}_hindi_summary.mp3",
                mime="audio/mp3"
            )
        else:
            st.error("Failed to generate Hindi audio. Please try again.")

if __name__ == "__main__":
    main()