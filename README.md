# News Sentiment Analysis with Hindi TTS

A web-based application that extracts news articles about a company, performs sentiment analysis, conducts comparative analysis, and generates Hindi text-to-speech output.

## Features

- **News Extraction**: Fetches up to 20 news articles related to a company using the News API
- **Sentiment Analysis**: Analyzes sentiment (positive, negative, neutral) of each article
- **Topic Extraction**: Identifies key topics discussed in each article
- **Comparative Analysis**: Conducts sentiment comparison across all articles
- **Data Visualization**: Provides interactive charts for sentiment distribution and topics
- **Hindi Text-to-Speech**: Converts analysis summary to Hindi audio
- **Downloadable Reports**: Generate CSV reports of analysis results
- **User-friendly Interface**: Simple Streamlit interface with tabs and expandable sections

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/news-sentiment-analysis.git
   cd news-sentiment-analysis
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your News API key (get one from [News API](https://newsapi.org/)):
   - Create a `.streamlit/secrets.toml` file with the following content:
     ```
     NEWS_API_KEY = "your_api_key_here"
     ```

## Usage

1. Start the Streamlit application:
   ```
   streamlit run main.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Either select a company from the dropdown or enter a custom company name

4. Click "Analyze News" to fetch and analyze articles

5. Explore the results in the three tabs:
   - Analysis: Overall sentiment metrics and visualizations
   - Articles: Detailed view of each article with sentiment and topics
   - Audio Summary: Hindi audio version of the analysis summary

## Technical Implementation

### Components

1. **News Extraction Module**
   - Uses News API to fetch articles
   - Implements web scraping with BeautifulSoup for full content extraction

2. **Sentiment Analysis Module**
   - Employs TextBlob for sentiment analysis
   - Calculates polarity and subjectivity scores
   - Classifies content as positive, negative, or neutral

3. **Topic Extraction Module**
   - Uses keyword-based approach to identify relevant topics
   - Categorizes articles into predefined topics

4. **Comparative Analysis Module**
   - Aggregates sentiment across articles
   - Generates insights on sentiment distribution
   - Analyzes sentiment by source and topic

5. **Text-to-Speech Module**
   - Converts analysis summary to Hindi speech using gTTS
   - Provides downloadable audio files

6. **Visualization Module**
   - Creates interactive charts with Matplotlib
   - Provides downloadable reports in CSV format

### API Workflow

1. User inputs company name
2. Frontend sends request to News API
3. Backend processes articles and performs sentiment analysis
4. Results are displayed in the Streamlit interface
5. Audio generation happens on-demand

## Deployment

This application is deployed on Hugging Face Spaces. You can access it at:
[https://huggingface.co/spaces/yourusername/news-sentiment-analysis](https://huggingface.co/spaces/yourusername/news-sentiment-analysis)

### Deployment Steps

1. Create an account on Hugging Face
2. Create a new Space with Streamlit SDK
3. Upload the application files
4. Add your News API key to the Space secrets

## Limitations and Future Improvements

- **Translation**: Currently, the application does not translate English to Hindi before TTS. A future version could incorporate translation APIs.
- **Advanced NLP**: Replace keyword-based topic extraction with more sophisticated NLP techniques like LDA.
- **Real-time Updates**: Implement WebSockets for real-time news updates.
- **User Accounts**: Add user authentication to save analysis history.
- **Custom Sentiment Models**: Train domain-specific sentiment models for better accuracy.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- News API for providing the news data
- Streamlit for the web framework
- TextBlob for sentiment analysis
- gTTS for Hindi text-to-speech conversion