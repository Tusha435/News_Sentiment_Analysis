# Stock Prediction Intelligence Platform

A production-grade, full-stack machine learning system that predicts stock price movement using hybrid intelligence from historical market data, company fundamentals, NLP-based sentiment analysis, reputation risk scoring, and TF-IDF text feature engineering.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    React Frontend (Vite + TailwindCSS)          │
│  ┌──────────┐ ┌──────────────┐ ┌────────────┐ ┌─────────────┐ │
│  │ Stock    │ │ Dual-Line    │ │ Sentiment  │ │ Reputation  │ │
│  │ Search   │ │ Prediction   │ │ Timeline   │ │ Gauge       │ │
│  │          │ │ Chart        │ │ Graph      │ │             │ │
│  └──────────┘ └──────────────┘ └────────────┘ └─────────────┘ │
│  ┌──────────────────────┐ ┌──────────────────────────────────┐ │
│  │ Volume vs Sentiment  │ │ News Impact Heatmap (TF-IDF)    │ │
│  │ Overlay              │ │                                  │ │
│  └──────────────────────┘ └──────────────────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────┘
                                │ REST API
┌───────────────────────────────┴─────────────────────────────────┐
│                    FastAPI Backend                               │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ Stock Data   │ │ Sentiment    │ │ Prediction Engine        │ │
│  │ Router       │ │ Router       │ │ Router                   │ │
│  └──────┬──────┘ └──────┬───────┘ └────────────┬─────────────┘ │
│         │               │                      │               │
│  ┌──────┴──────┐ ┌──────┴───────┐ ┌────────────┴─────────────┐ │
│  │ Alpha       │ │ Sentiment    │ │ ML Models                │ │
│  │ Vantage     │ │ Analyzer     │ │ ┌────────┐ ┌──────────┐  │ │
│  │ Client      │ │ (VADER+TB)   │ │ │XGBoost │ │Random    │  │ │
│  ├─────────────┤ ├──────────────┤ │ │        │ │Forest    │  │ │
│  │ News        │ │ TF-IDF       │ │ └────────┘ └──────────┘  │ │
│  │ Collector   │ │ Engine       │ │ ┌────────┐ ┌──────────┐  │ │
│  ├─────────────┤ ├──────────────┤ │ │LSTM    │ │Hybrid    │  │ │
│  │ Technical   │ │ Reputation   │ │ │        │ │Ensemble  │  │ │
│  │ Indicators  │ │ Scorer       │ │ └────────┘ └──────────┘  │ │
│  └─────────────┘ └──────────────┘ └──────────────────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │            Feature Engineering Layer                      │   │
│  │  Structured: OHLCV, SMA, EMA, RSI, MACD, Bollinger, ATR │   │
│  │  Unstructured: Sentiment, Reputation, TF-IDF, News Freq  │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │   SQLite / PostgreSQL  │
                    └───────────────────────┘
```

## Features

### Data Ingestion Layer
- **Alpha Vantage API**: Historical OHLCV, intraday, company fundamentals, earnings
- Rate-limit handling (12s between calls for free tier)
- Response caching (1-hour TTL)
- Retry logic with exponential backoff

### Sentiment Intelligence Engine
- **VADER**: Augmented with 30+ financial domain terms (bullish, bearish, bankruptcy, etc.)
- **TextBlob**: General-purpose NLP baseline
- **Ensemble Model**: 60% VADER + 40% TextBlob weighted combination
- Named Entity Recognition (NER) for company/person extraction
- Text preprocessing: stopword removal, lemmatization, URL cleanup

### Reputation Risk Scoring (SIS Formula)

```
SIS = Sentiment_Polarity x Event_Weight x News_Reach x Recency_Decay
```

| Component | Description | Range |
|-----------|-------------|-------|
| Sentiment_Polarity | Ensemble sentiment score | [-1, 1] |
| Event_Weight | Category-specific weight (bankruptcy=-3.0, profit_surge=+2.5, etc.) | [-3, 3] |
| News_Reach | Source credibility score (Reuters=1.0, blogs=0.5) | [0, 1] |
| Recency_Decay | Exponential decay: `e^(-lambda * days_old)`, half-life=7 days | [0, 1] |

**Event categories**: bankruptcy, fraud, mass_layoffs, legal_action, CEO departure, downgrade, earnings_miss, profit_surge, product_launch, upgrade, acquisition, expansion, dividend, partnership

**Risk levels**: LOW (75-100), MEDIUM (50-75), HIGH (25-50), CRITICAL (0-25)

### TF-IDF Importance Modeling
- Custom financial vocabulary (80+ terms)
- Unigram + bigram feature extraction
- Sublinear TF scaling (`1 + log(tf)`)
- Per-article and corpus-level term importance
- Feature vector generation for ML pipeline

### Technical Indicators
| Indicator | Parameters |
|-----------|-----------|
| SMA | 5, 10, 20, 50, 200 |
| EMA | 12, 26 |
| RSI | 14-period |
| MACD | 12/26/9 |
| Bollinger Bands | 20-period, 2 std |
| ATR | 14-period |
| OBV | On-Balance Volume |
| VWAP | Volume Weighted Avg Price |
| Momentum | 5, 10, 20-day |
| Volatility | 10, 20-day rolling std |

### ML Models

| Model | Type | Purpose |
|-------|------|---------|
| Ridge Regression | Baseline | Price prediction |
| Random Forest | Ensemble (200 trees) | Price + direction |
| XGBoost | Gradient boosting (300 rounds) | Price + direction |
| LSTM | Recurrent neural network (64 hidden) | Sequential patterns |
| **Hybrid** | Inverse-RMSE weighted ensemble | Combined prediction |

**Training pipeline**: TimeSeriesSplit cross-validation (5 folds), early stopping, hyperparameter tuning.

**Hybrid ensemble weights**: Determined automatically by inverse validation RMSE of each component model.

### Prediction Output
- **Next-day** (1d): Short-term price + direction + confidence
- **Weekly** (5d): Medium-term trend
- **Monthly** (21d): Long-term projection

### Frontend Dashboard
- Dual-line chart: actual market price vs AI predicted price
- Sentiment timeline with news count overlay
- Reputation risk semicircular gauge
- TF-IDF news impact heatmap
- Volume vs sentiment overlay chart
- Prediction cards with direction arrows and confidence scores
- Technical indicator summary bar
- News article feed with sentiment badges

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI application
│   │   ├── config.py               # Configuration
│   │   ├── models/
│   │   │   ├── database.py         # SQLAlchemy models
│   │   │   └── schemas.py          # Pydantic schemas
│   │   ├── services/
│   │   │   ├── alpha_vantage.py    # Stock data API client
│   │   │   ├── news_service.py     # News collection
│   │   │   ├── sentiment.py        # Sentiment analysis
│   │   │   ├── tfidf_engine.py     # TF-IDF features
│   │   │   ├── reputation.py       # Reputation risk scoring
│   │   │   ├── technical.py        # Technical indicators
│   │   │   ├── features.py         # Feature engineering
│   │   │   ├── ml_models.py        # ML model implementations
│   │   │   └── prediction.py       # Prediction orchestrator
│   │   └── routers/
│   │       ├── stocks.py           # Stock data endpoints
│   │       ├── predictions.py      # Prediction endpoints
│   │       └── sentiment_router.py # Sentiment endpoints
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # Main application
│   │   ├── main.jsx                # Entry point
│   │   ├── index.css               # TailwindCSS styles
│   │   ├── components/
│   │   │   ├── Dashboard.jsx       # Main dashboard layout
│   │   │   ├── StockSearch.jsx     # Search bar
│   │   │   ├── PredictionChart.jsx # Dual-line price chart
│   │   │   ├── SentimentTimeline.jsx
│   │   │   ├── ReputationGauge.jsx # Risk score gauge
│   │   │   ├── NewsImpact.jsx      # News + TF-IDF heatmap
│   │   │   └── VolumeSentimentOverlay.jsx
│   │   └── services/
│   │       └── api.js              # API client
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── Dockerfile
├── models/
│   ├── train.py                    # Model training script
│   └── evaluate.py                 # Model evaluation script
├── data_pipeline/
│   ├── collect_data.py             # Data collection pipeline
│   └── process_data.py             # Data processing pipeline
├── tests/
│   ├── test_api.py                 # API endpoint tests
│   ├── test_sentiment.py           # Sentiment analysis tests
│   ├── test_technical.py           # Technical indicator tests
│   ├── test_reputation.py          # Reputation scoring tests
│   ├── test_tfidf.py               # TF-IDF engine tests
│   └── test_models.py              # ML model tests
├── deployment/
│   ├── docker-compose.yml
│   └── .env.example
├── main.py                         # Original Streamlit app
└── README.md
```

## Setup Instructions

### Prerequisites
- Python 3.11+
- Node.js 18+
- API keys: [Alpha Vantage](https://www.alphavantage.co/support/) (free) and [NewsAPI](https://newsapi.org/) (free)

### 1. Clone and configure

```bash
git clone https://github.com/Tusha435/News_Sentiment_Analysis.git
cd News_Sentiment_Analysis
cp .env.example .env
# Edit .env with your API keys
```

### 2. Backend setup

```bash
pip install -r backend/requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('maxent_ne_chunker'); nltk.download('maxent_ne_chunker_tab'); nltk.download('words')"
```

### 3. Start the backend

```bash
uvicorn backend.app.main:app --reload --port 8000
```

### 4. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

### 5. Open the dashboard

Navigate to `http://localhost:3000` and enter a stock symbol (e.g., AAPL).

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stocks/search?q=AAPL` | Search stock symbols |
| GET | `/api/stocks/{symbol}/prices?days=365` | Get prices + technical indicators |
| GET | `/api/stocks/{symbol}/overview` | Company profile |
| GET | `/api/stocks/{symbol}/fundamentals` | Financial statements |
| GET | `/api/sentiment/{symbol}/analyze` | Sentiment analysis |
| POST | `/api/sentiment/analyze-text` | Analyze arbitrary text |
| GET | `/api/predictions/{symbol}` | Get predictions |
| POST | `/api/predictions/train` | Train model |
| GET | `/api/predictions/{symbol}/dashboard` | Full dashboard data |
| GET | `/docs` | Interactive API docs (Swagger) |

## CLI Tools

### Train a model
```bash
python -m models.train --symbol AAPL --model hybrid --days 730
```

### Evaluate a model
```bash
python -m models.evaluate --symbol AAPL --model hybrid --test-days 60
```

### Collect data
```bash
python -m data_pipeline.collect_data --symbol AAPL --days 730
```

### Process data
```bash
python -m data_pipeline.process_data --symbol AAPL
```

## Docker Deployment

```bash
cd deployment
cp .env.example .env
# Edit .env with your API keys
docker compose up --build
```

Backend: `http://localhost:8000` | Frontend: `http://localhost:3000`

## Running Tests

```bash
python -m pytest tests/ -v
```

## Data Flow

```
Alpha Vantage API ──> OHLCV Data ──> Technical Indicators ──┐
                                                             │
NewsAPI / RSS ──> Article Text ──> Sentiment (VADER+TB) ──>  │
                       │                                     │
                       ├──> TF-IDF Features ────────────>   Feature  ──> ML Models ──> Predictions
                       │                                     Matrix      (XGB+RF+LSTM)
                       └──> Reputation Score (SIS) ──────>   │
                                                             │
Company Fundamentals ────────────────────────────────────>   │
```
