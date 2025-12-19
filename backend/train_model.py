# train_model.py
# Script to train the recommendation model from raw data

import pandas as pd
import pickle
import os
from recommender import create_tfidf_vectorizer, fit_and_transform, clean_text
from config import (
    RAW_REVIEWS_FILE,
    PROCESSED_GAMES_FILE,
    VECTORIZER_PATH,
    GAME_VECTORS_PATH,
    MIN_REVIEWS_PER_GAME,
    TFIDF_CONFIG
)


def process_data():
    """Load and process raw review data"""
    print("="*60)
    print("Step 1: Loading and Processing Data")
    print("="*60)
    
    # Load raw reviews in chunks for large files
    print(f"\nLoading raw data from {RAW_REVIEWS_FILE}...")
    print("(Reading first 500,000 reviews to get more games...)")
    
    # Read first 500k rows to get more games (should give 100+ games)
    df = pd.read_csv(RAW_REVIEWS_FILE, nrows=500000)
    print(f"✓ Loaded {len(df)} reviews")
    print(f"✓ Unique games: {df['app_id'].nunique()}")
    
    # Clean review texts
    print("\nCleaning review texts...")
    df['cleaned_review'] = df['review_text'].apply(clean_text)
    
    # Remove empty reviews
    df = df[df['cleaned_review'].str.strip() != '']
    print(f"✓ {len(df)} reviews after cleaning")
    
    # Group by game and combine reviews
    print("\nGrouping reviews by game...")
    game_data = df.groupby(['app_id', 'app_name']).agg({
        'cleaned_review': lambda x: ' '.join(x),
        'review_text': 'count'
    }).reset_index()
    
    game_data.columns = ['app_id', 'app_name', 'combined_reviews', 'review_count']
    
    # Filter games with minimum reviews
    game_data = game_data[game_data['review_count'] >= MIN_REVIEWS_PER_GAME]
    print(f"✓ {len(game_data)} games with at least {MIN_REVIEWS_PER_GAME} reviews")
    
    # Save processed data
    game_data[['app_id', 'app_name', 'review_count']].to_csv(PROCESSED_GAMES_FILE, index=False)
    print(f"✓ Saved processed game data to {PROCESSED_GAMES_FILE}")
    
    return game_data


def train_model(game_data):
    """Train TF-IDF model on game reviews"""
    print("\n" + "="*60)
    print("Step 2: Training TF-IDF Model")
    print("="*60)
    
    # Get documents
    documents = game_data['combined_reviews'].tolist()
    
    # Create and train vectorizer
    print("\nCreating TF-IDF vectorizer...")
    vectorizer = create_tfidf_vectorizer()
    
    print("Training on game reviews...")
    game_vectors = fit_and_transform(vectorizer, documents)
    
    # Save vectorizer
    print(f"\nSaving vectorizer to {VECTORIZER_PATH}...")
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    print("✓ Vectorizer saved")
    
    # Save game vectors
    print(f"Saving game vectors to {GAME_VECTORS_PATH}...")
    with open(GAME_VECTORS_PATH, 'wb') as f:
        pickle.dump(game_vectors, f)
    print("✓ Game vectors saved")
    
    return vectorizer, game_vectors


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("  Steam Game Recommendation Model Training")
    print("="*60 + "\n")
    
    try:
        # Process data
        game_data = process_data()
        
        # Train model
        vectorizer, game_vectors = train_model(game_data)
        
        # Summary
        print("\n" + "="*60)
        print("  Training Complete!")
        print("="*60)
        print(f"✓ Total games: {len(game_data)}")
        print(f"✓ Vocabulary size: {len(vectorizer.vocabulary_)}")
        print(f"✓ Vector dimensions: {game_vectors.shape}")
        print("\nModel files saved:")
        print(f"  - {VECTORIZER_PATH}")
        print(f"  - {GAME_VECTORS_PATH}")
        print(f"  - {PROCESSED_GAMES_FILE}")
        print("\nYou can now run the API server:")
        print("  python app.py")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
