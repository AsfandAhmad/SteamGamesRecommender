# Steam Game Recommender 🎮

A TF-IDF based game recommendation system that suggests Steam games based on user preferences and reviews.

## Features
- **Smart Recommendations**: Uses TF-IDF and Cosine Similarity to find matching games
- **Profile-Based Ranking**: Generate top 100 game recommendations
- **Interactive UI**: Beautiful Streamlit interface
- **Flask API**: RESTful API for integrations

## Project Structure
```
steam-recommender/
├── backend/
│   ├── recommender/         # Core recommendation modules
│   ├── models/              # Trained ML models
│   ├── app.py              # Flask API server
│   ├── profile.py          # Profile-based recommendations
│   ├── train_model.py      # Model training script
│   └── requirements.txt    # Dependencies
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Processed data
└── streamlit_app.py        # Streamlit frontend
```

## Setup & Installation

### 1. Install Dependencies
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt

# Install Streamlit (for frontend)
pip install streamlit
```

### 2. Train the Model
```bash
# Make sure you're in the backend directory with venv activated
cd backend
source venv/bin/activate

# Train the recommendation model (uses first 100k reviews)
python train_model.py
```

This will:
- Load and clean the raw review data
- Train the TF-IDF vectorizer
- Save models to `backend/models/`
- Save processed data to `data/processed/`

## Running the Project

### Option 1: Streamlit Frontend (Recommended)
```bash
# From project root directory
cd /home/asfand-ahmed/Desktop/SteamGames_RS/steam-recommender

# Run Streamlit app
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

**Features:**
- Search by review/preferences
- Generate top 100 ranked games
- Genre filters
- Download results as CSV

### Option 2: Flask API
```bash
# Navigate to backend with venv activated
cd backend
source venv/bin/activate

# Start Flask server
python app.py
```

Server runs at `http://localhost:5000`

**API Endpoints:**
- `GET /` - Health check
- `GET /api/games` - List all games
- `POST /api/recommend` - Get recommendations
- `GET /api/stats` - System statistics

**Example API Request:**
```bash
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"review": "action games with great graphics", "top_n": 10}'
```

### Option 3: Profile-Based CLI
```bash
# Navigate to backend
cd backend
source venv/bin/activate

# Run profile recommendations
python profile.py
```

Interactive CLI that:
- Takes your gaming preferences
- Generates top 100 recommendations
- Saves results to CSV

## Usage Examples

### Streamlit App
1. Open the app in browser
2. **Tab 1 - Search by Review**: 
   - Enter "I love fast-paced action shooters"
   - Click "Get Recommendations"
   - See ranked results
3. **Tab 2 - Top 100 Games**:
   - Enter profile text
   - Select genres (optional)
   - Generate comprehensive rankings

### Flask API
```python
import requests

response = requests.post('http://localhost:5000/api/recommend', 
    json={
        'review': 'multiplayer cooperative adventure games',
        'top_n': 5
    }
)

print(response.json())
```

## Dataset
- Source: Steam reviews dataset (2.1GB)
- Location: `data/raw/steam_reviews.csv`
- Processed: First 100,000 reviews (39 games)
- Columns: `app_id`, `app_name`, `review_text`, `review_score`, `review_votes`

## Technical Details
- **Algorithm**: TF-IDF Vectorization + Cosine Similarity
- **Vocabulary**: 5000 most important words
- **N-grams**: Unigrams and bigrams
- **Preprocessing**: Lowercase, remove punctuation, remove stopwords, lemmatization

## Requirements
- Python 3.8+
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- nltk >= 3.8.0
- flask >= 3.0.0
- flask-cors >= 4.0.0
- streamlit (for frontend)

## Troubleshooting

### Model not found
```bash
# Train the model first
cd backend
source venv/bin/activate
python train_model.py
```

### Module not found
```bash
# Install dependencies
cd backend
source venv/bin/activate
pip install -r requirements.txt
pip install streamlit
```

### NLTK data not found
The first run will automatically download required NLTK data (stopwords, wordnet).

## Quick Start Commands
```bash
# Complete setup and run
cd /home/asfand-ahmed/Desktop/SteamGames_RS/steam-recommender/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install streamlit
python train_model.py
cd ..
streamlit run streamlit_app.py
```

---
**Powered by TF-IDF & Cosine Similarity** 🚀
