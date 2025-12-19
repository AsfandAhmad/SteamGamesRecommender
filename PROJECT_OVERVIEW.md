# Steam Game Recommender System - Project Overview

## 🎯 Project Purpose
An AI-powered recommendation system that analyzes Steam game reviews using Natural Language Processing (NLP) to suggest games based on user preferences.

---

## 📁 Project Structure

```
steam-recommender/
├── backend/
│   ├── recommender/         # Core recommendation modules
│   ├── models/              # Trained ML models
│   ├── data/                # Dataset storage
│   ├── app.py               # Flask REST API
│   ├── train_model.py       # Model training pipeline
│   ├── profile.py           # CLI recommendations
│   └── config.py            # Configuration settings
├── streamlit_app.py         # Web UI frontend
└── data/raw/                # Raw dataset
```

---

## 🔧 Core Files Explanation

### **1. backend/recommender/preprocess.py**
**Purpose:** Text cleaning and preprocessing  
**How it works:**
- Takes raw review text as input
- Converts to lowercase
- Removes punctuation and special characters using regex
- Filters out stopwords (common words like "the", "is", "and")
- Lemmatizes words (converts to root form: "running" → "run")
- Returns clean text ready for vectorization

**Key Function:** `clean_text(text)` → Returns cleaned string

---

### **2. backend/recommender/vectorizer.py**
**Purpose:** Convert text to numerical vectors  
**How it works:**
- Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** algorithm
- TF-IDF measures word importance in documents
- Creates vectors with 5000 features (most important words)
- Parameters:
  - `max_features=5000` - Top 5000 words
  - `min_df=2` - Word must appear in at least 2 games
  - `max_df=0.8` - Ignore words in >80% of games
  - `ngram_range=(1,2)` - Single words + word pairs

**Key Functions:**
- `create_tfidf_vectorizer()` → Creates vectorizer
- `fit_and_transform(texts)` → Trains on data
- `transform_text(text)` → Converts new text to vector

---

### **3. backend/recommender/similarity.py**
**Purpose:** Calculate similarity between games  
**How it works:**
- Uses **Cosine Similarity** to compare vectors
- Cosine similarity measures angle between vectors (0-1 scale)
- Closer to 1 = more similar
- Finds games with highest similarity scores

**Key Functions:**
- `calculate_similarity_matrix(vectors)` → Creates similarity matrix
- `find_top_matches(vector, n)` → Returns top N similar games
- `get_similarity_scores(query, games)` → Gets scores for query

---

### **4. backend/recommender/recommend.py**
**Purpose:** Main recommendation engine  
**How it works:**
1. User enters preferences (e.g., "action shooter multiplayer")
2. Preprocesses user input using `clean_text()`
3. Converts to vector using trained TF-IDF vectorizer
4. Calculates similarity with all game vectors
5. Returns top N games ranked by similarity

**Key Class:** `GameRecommender`
- `recommend(user_input, top_n)` → Returns recommendations list

**Output Format:**
```python
[
  {
    'game_id': 123,
    'game_name': 'Counter-Strike',
    'similarity_score': 0.87,
    'match_percentage': 87.0
  },
  ...
]
```

---

### **5. backend/config.py**
**Purpose:** Centralized configuration  
**What it contains:**
- File paths (data, models, processed files)
- TF-IDF parameters (max_features, min_df, etc.)
- Flask API settings (host, port)
- Model names and locations

**Why important:** Single place to change settings across entire project

---

### **6. backend/train_model.py**
**Purpose:** Train the recommendation model  
**How it works:**

**Step 1: Load Data**
- Reads `steam_reviews.csv` (500,000 reviews)
- Columns: app_id, app_name, review_text, review_score, review_votes

**Step 2: Clean Reviews**
- Applies `clean_text()` to all review texts
- Handles NaN values and errors

**Step 3: Aggregate by Game**
- Groups reviews by game (app_id)
- Combines all reviews for each game into one text
- Result: One document per game

**Step 4: Train TF-IDF**
- Creates TF-IDF vectorizer
- Fits on all game documents
- Generates vectors for each game

**Step 5: Save Models**
- Saves `tfidf_vectorizer.pkl` (the trained vectorizer)
- Saves `game_vectors.pkl` (game vector matrix)
- Saves `games_aggregated.csv` (game metadata)

**Output:** 332 games with trained models

---

### **7. backend/app.py**
**Purpose:** REST API for recommendations  
**How it works:**

**Endpoints:**
1. **GET /** - Health check
2. **GET /api/games** - List all games
3. **POST /api/recommend** - Get recommendations
   - Input: `{"review_text": "user preferences", "top_n": 10}`
   - Output: JSON with recommendations
4. **GET /api/stats** - System statistics

**Technology:** Flask with CORS enabled for frontend access

---

### **8. backend/profile.py**
**Purpose:** CLI-based recommendation tool  
**How it works:**
- User provides gaming profile via command line
- Generates top 100 recommendations
- Categorizes into tiers:
  - 🔥 Highly Recommended (1-10)
  - ⭐ Recommended (11-50)
  - 💡 Least Recommended (51-100)
- Saves results to `user_recommendations.csv`

**Usage:** `python profile.py`

---

### **9. streamlit_app.py**
**Purpose:** Web UI for end users  
**How it works:**

**Tab 1: Search by Preferences**
1. User enters gaming preferences in text area
2. Clicks "Find My Perfect Games"
3. System processes input and shows top N recommendations
4. Results displayed with match percentages and styling
5. Download button for CSV export

**Tab 2: Top 100 Rankings**
1. User creates gaming profile
2. Selects preferred genres (checkboxes)
3. Clicks "Generate Top 100"
4. Shows rankings in 3 sections:
   - TOP 10 (Must Play) - Featured cards
   - 11-50 (Recommended) - Data table
   - 51-100 (Explore More) - Data table
5. Download button for full list

**UI Features:**
- Steam-themed blue color scheme (#1b2838, #66c0f4, #1A9FFF)
- Gradient backgrounds and glowing effects
- Responsive game cards with hover animations
- Custom CSS for gaming aesthetic
- Sidebar with settings and stats

**Caching:** `@st.cache_resource` loads models once for performance

---

## 🔄 Complete Workflow

### **Training Phase (One-time Setup)**
```
1. Load steam_reviews.csv (500K reviews)
   ↓
2. Clean all review texts (NLP preprocessing)
   ↓
3. Group reviews by game (332 unique games)
   ↓
4. Train TF-IDF vectorizer on game documents
   ↓
5. Generate vector representation for each game
   ↓
6. Save models: vectorizer.pkl, vectors.pkl, games.csv
```

### **Recommendation Phase (User Request)**
```
1. User enters: "fast-paced action shooter with multiplayer"
   ↓
2. Clean user input (preprocess.py)
   ↓
3. Convert to vector using trained vectorizer (vectorizer.py)
   ↓
4. Calculate cosine similarity with all 332 game vectors (similarity.py)
   ↓
5. Rank games by similarity score (recommend.py)
   ↓
6. Return top N games with match percentages
   ↓
7. Display in UI (streamlit_app.py) with styling
```

---

## 🧠 Machine Learning Approach

### **Algorithm: TF-IDF + Cosine Similarity**

**Why TF-IDF?**
- Captures word importance in documents
- Reduces impact of common words
- Highlights distinctive game features
- Works well with text-based recommendations

**Why Cosine Similarity?**
- Measures semantic similarity between texts
- Range: 0 (completely different) to 1 (identical)
- Fast computation for real-time recommendations
- Effective for high-dimensional sparse data

**Mathematical Flow:**
```
Review Text → TF-IDF Vector → Cosine Similarity → Match Score
"action shooter" → [0.2, 0.8, ...] → cos(θ) = 0.87 → 87% match
```

---

## 📊 Data Flow

```
Raw Data (steam_reviews.csv)
         ↓
  Preprocessing (clean_text)
         ↓
  Aggregation (group by game)
         ↓
  Vectorization (TF-IDF)
         ↓
  Model Storage (PKL files)
         ↓
  User Query → Vectorize → Similarity → Results
         ↓
  Display (Streamlit UI)
```

---

## 🎨 UI/UX Design Approach

**Theme:** Steam Gaming Platform
- Dark blue gradients (Steam brand colors)
- Glowing effects on interactive elements
- Professional gaming aesthetic
- Smooth animations and transitions

**User Experience:**
- Simple text input (no complex forms)
- Quick examples for inspiration
- Visual feedback (loading spinners, success messages)
- Categorized results (Top 10 featured prominently)
- Export functionality (CSV downloads)

**Responsive Design:**
- Column layouts for different screen sections
- Centered buttons for focus
- Card-based game displays
- Collapsible data tables for large lists

---

## 🚀 Technology Stack

**Backend:**
- Python 3.12
- scikit-learn (ML algorithms)
- NLTK (NLP processing)
- pandas (data manipulation)
- Flask (REST API)
- pickle (model serialization)

**Frontend:**
- Streamlit (web framework)
- Custom CSS (Steam theme)
- HTML/Markdown (content)

**Data Processing:**
- NumPy (numerical operations)
- SciPy (sparse matrices)

---

## 📈 Performance Metrics

**Model Statistics:**
- Training data: 500,000 reviews
- Unique games: 332
- Vocabulary size: 5,000 features
- Vector dimensions: 5,000
- Training time: ~10-15 minutes
- Recommendation time: <1 second

**Accuracy Approach:**
- Content-based filtering (not collaborative)
- Similarity threshold: Scores above 40% shown
- Top 10 results typically >70% match

---

## 🔮 How It Works (Simple Explanation)

1. **Training:** System reads 500K reviews, learns which words describe which games, creates a "fingerprint" (vector) for each of 332 games

2. **User Input:** You describe your ideal game (e.g., "scary horror survival")

3. **Matching:** System converts your description to a fingerprint, compares with all 332 game fingerprints

4. **Results:** Games with most similar fingerprints are recommended with match percentages

5. **Display:** Beautiful UI shows results ranked from best to good matches

**Think of it like:** Fingerprint matching for games based on review text patterns

---

## 🎓 Key Concepts

**TF-IDF:** Weighs words by how unique they are to a game  
**Cosine Similarity:** Measures angle between text vectors  
**Vectorization:** Converting text to numbers for math  
**NLP:** Teaching computers to understand human language  
**Content-Based Filtering:** Recommending similar items based on features  

---

## 💡 Project Strengths

✅ Fast real-time recommendations  
✅ No user history needed (cold start friendly)  
✅ Explainable results (based on text similarity)  
✅ Scalable to more games  
✅ Beautiful, intuitive UI  
✅ Multiple interfaces (API, CLI, Web)  
✅ Professional Steam-themed design  

---

## 📝 Future Enhancements

- Add collaborative filtering (user-based recommendations)
- Include game metadata (genre, price, ratings)
- Real-time data updates from Steam API
- User accounts and saved preferences
- Game images and trailers in results
- Multi-language support
- More detailed similarity explanations

---

**Built with ❤️ using AI & Machine Learning**
