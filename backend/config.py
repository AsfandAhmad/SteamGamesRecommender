# config.py
# This file contains all configuration settings for the backend
# Keeps all settings in one place for easy management

import os

# Base directory - where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# File paths
RAW_REVIEWS_FILE = os.path.join(RAW_DATA_DIR, 'steam_reviews.csv')
PROCESSED_GAMES_FILE = os.path.join(PROCESSED_DATA_DIR, 'games_aggregated.csv')

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, 'models')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
GAME_VECTORS_PATH = os.path.join(MODEL_DIR, 'game_vectors.pkl')

# TF-IDF Vectorizer settings
TFIDF_CONFIG = {
    'max_features': 5000,      # Maximum number of words to keep
    'min_df': 2,               # Minimum document frequency
    'max_df': 0.8,             # Maximum document frequency
    'ngram_range': (1, 2)      # Use single words and two-word phrases
}

# Recommendation settings
DEFAULT_TOP_N = 5              # Default number of recommendations to return
MAX_TOP_N = 20                 # Maximum number of recommendations allowed

# Flask app settings
FLASK_HOST = '0.0.0.0'         # Host to run the Flask app
FLASK_PORT = 5000              # Port to run the Flask app
FLASK_DEBUG = True             # Debug mode (set to False in production)

# CORS settings
CORS_ORIGINS = '*'             # Allow requests from any origin (change in production)

# Data processing settings
SAMPLE_SIZE = 50000            # Number of reviews to use for training
MIN_REVIEWS_PER_GAME = 5       # Minimum reviews a game needs to be included

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
