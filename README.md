# 🎬 FlixVerse — Movie Recommendation System

A production-ready, AI-powered movie recommendation engine built with Python, Pandas, scikit-learn, and Streamlit.

## ✨ Features

- **ML Engine** — TF-IDF vectorization + Cosine Similarity for content-based recommendations
- **Live Posters** — Fetched from The Movie Database (TMDb) API
- **Smart Search** — Real-time title search with instant filtering
- **Advanced Filters** — Filter by genre, year range, and minimum IMDb rating
- **Genre Explorer** — Browse top-rated movies by genre
- **Flixverse UI** — Dark, editorial design with Playfair Display typography

## 🧠 How the ML Works

1. **Feature Engineering** — Combines genres, director name, and plot overview into a single text feature per movie
2. **TF-IDF Vectorization** — Converts text features into numerical vectors (5000 features, English stopwords removed)
3. **Cosine Similarity Matrix** — Computes pairwise similarity scores between all 50+ movies
4. **Ranking** — Sorts by similarity score, then applies user-defined filters

## 🚀 Run Locally

```bash
# Clone / download the project
cd movie_recommender

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this folder to a GitHub repository
2. Go to https://streamlit.io/cloud and sign in
3. Click **New app** → select your repo → set `app.py` as the main file
4. Click **Deploy** — live in ~2 minutes!

## 🌐 Deploy to Other Platforms

### Railway
```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
railway init
railway up
```

### Render
- Create new **Web Service** on render.com
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.headless true`

### Hugging Face Spaces
- Create a new Space with **Streamlit** SDK
- Upload all files — it deploys automatically

## 📁 Project Structure

```

```

## 🔧 Extend It

- **More movies** — Add rows to the `movies` list in `load_data()`
- **Collaborative filtering** — Add user ratings and use SVD/ALS
- **Real dataset** — Swap in the full MovieLens dataset (25M ratings)
- **Watchlist** — Use `st.session_state` to track saved movies

## 📊 Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| ML | scikit-learn (TF-IDF, Cosine Similarity) |
| Data | Pandas, NumPy |
| Posters | TMDb REST API |
| Deploy | Streamlit Cloud / Railway / Render |
