# TRAVELMIND ✈️

> AI-powered travel assistant built with RAG (Retrieval-Augmented Generation) — real-time flight data, smart analytics, and intelligent recommendations.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3-orange?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-purple?style=flat-square)

---

## What is TravelMind?

TravelMind is a RAG-powered travel planning assistant that lets users search for flights using natural language, get AI-generated answers based on real flight data, and explore destination guides — all in one place.

Instead of filling out forms like traditional flight search engines, you simply ask:

> *"Which flights are available from Kochi to Delhi today?"*
> *"Is there a direct flight to Singapore this weekend?"*

TravelMind fetches real-time flight data, embeds it into a vector database, retrieves the most relevant results, and passes them to an LLM to generate a clear, helpful answer.

---

## Features

- **Natural Language Flight Search** — Ask questions in plain English instead of filling forms
- **RAG Pipeline** — Real flight data retrieved from vector store and passed to LLM as context
- **Real-time Weather** — Live weather at destination powered by OpenWeatherMap API
- **Price Analytics** — Price trend charts, best day to fly heatmap, and price prediction
- **Destination Guide** — AI-generated top places, hotels, and foods for any destination
- **Group Travel Planner** — Find flights for multiple travellers flying from different cities
- **Explainable AI** — Answers are grounded in real retrieved flight data, not hallucinated

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| Frontend | Streamlit |
| RAG Framework | LangChain |
| Vector Store | ChromaDB |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| LLM | Groq API (LLaMA 3.3 70B) |
| Flight Data | Aviationstack API |
| Weather Data | OpenWeatherMap API |
| Database | SQLite |
| Charts | Plotly |

---

## Architecture

```
User Query
    ↓
Aviationstack API → Real Flight Data
    ↓
SentenceTransformers → Embed flight text into vectors
    ↓
ChromaDB → Store and retrieve relevant flight vectors
    ↓
Groq LLM (LLaMA 3.3) → Generate natural language answer
    ↓
Streamlit → Display to user
```

---

## Project Structure

```
travelmind/
├── app.py              # Main Streamlit application
├── database.py         # SQLite price history functions
├── .env                # API keys (not committed)
├── .env.example        # Example env file
├── .gitignore          # Git ignore rules
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/avanthikasreejith8923-a11y/travelmind.git
cd travelmind
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API keys

Create a `.env` file in the root folder:

```
AVIATIONSTACK_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
OPENWEATHER_API_KEY=your_key_here
```

Get your free API keys from:
- Aviationstack: https://aviationstack.com
- Groq: https://console.groq.com
- OpenWeatherMap: https://openweathermap.org/api

### 5. Run the app

```bash
streamlit run app.py
```

---

## How RAG Works Here

Traditional chatbots rely on training data which gets outdated. TravelMind uses RAG to solve this:

1. **Retrieve** — When you search, live flight data is fetched from Aviationstack API
2. **Augment** — That data is converted to text, embedded, and stored in ChromaDB
3. **Generate** — Your question + retrieved flight context is sent to LLaMA 3.3, which generates a grounded answer

This means the AI never makes up flight information — it only answers based on real data fetched at query time.

---

## API Keys Required

| API | Free Tier | Usage |
|-----|-----------|-------|
| Aviationstack | 500 calls/month | Real-time flight data |
| Groq | Generous free tier | LLM inference |
| OpenWeatherMap | 1000 calls/day | Destination weather |

---

## Disclaimer

Price analytics use estimated fares based on typical route patterns for demonstration purposes. For actual booking prices, please check airline websites directly.

---






















