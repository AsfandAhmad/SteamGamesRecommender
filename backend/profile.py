# profile.py
# Generate game recommendations based on user preferences and genres

import pandas as pd
import pickle
from recommender import GameRecommender, clean_text
from config import VECTORIZER_PATH, GAME_VECTORS_PATH, PROCESSED_GAMES_FILE, RAW_REVIEWS_FILE


def load_game_data_with_genres():
    """Load game data and extract genres from reviews"""
    print("Loading game data...")
    
    # Load processed games
    games_df = pd.read_csv(PROCESSED_GAMES_FILE)
    
    # Load some raw reviews to get genre information
    raw_df = pd.read_csv(RAW_REVIEWS_FILE, nrows=100000)
    
    # Merge to get game names
    games_with_reviews = raw_df.merge(games_df[['app_id', 'app_name']], on='app_id', how='inner')
    
    return games_df, games_with_reviews


def create_user_profile(preferences, genres=None):
    """
    Create a user profile based on preferences and genres.
    
    Args:
        preferences: User's text description of what they like
        genres: List of preferred genres (optional)
    
    Returns:
        Combined text representing user profile
    """
    profile_text = preferences
    
    if genres:
        genre_text = " ".join(genres)
        profile_text = f"{preferences} {genre_text}"
    
    return profile_text


def get_top_recommendations(user_profile, top_n=100):
    """
    Get top N game recommendations based on user profile.
    
    Args:
        user_profile: User's preference text
        top_n: Number of recommendations to return (default 100)
    
    Returns:
        List of recommended games with scores
    """
    print(f"\nGenerating top {top_n} recommendations...")
    
    # Load vectorizer and game vectors
    print("Loading models...")
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(GAME_VECTORS_PATH, 'rb') as f:
        game_vectors = pickle.load(f)
    
    # Load game data
    games_df = pd.read_csv(PROCESSED_GAMES_FILE)
    game_names = games_df['app_name'].tolist()
    game_ids = games_df['app_id'].tolist()
    
    # Create recommender
    recommender = GameRecommender(vectorizer, game_vectors, game_names, game_ids)
    
    # Get recommendations
    recommendations = recommender.recommend(user_profile, top_n=top_n)
    
    # Add ranking
    for i, rec in enumerate(recommendations, 1):
        rec['rank'] = i
        rec['recommendation_level'] = 'Highly Recommended' if i <= 10 else 'Recommended' if i <= 50 else 'Least Recommended'
    
    return recommendations


def display_recommendations(recommendations):
    """Display recommendations in a formatted way"""
    print("\n" + "="*80)
    print("  TOP GAME RECOMMENDATIONS")
    print("="*80 + "\n")
    
    # Top 10
    print("🔥 TOP 10 - HIGHLY RECOMMENDED:")
    print("-" * 80)
    for rec in recommendations[:10]:
        print(f"{rec['rank']:3d}. {rec['game_name']:<50s} Match: {rec['match_percentage']:5.1f}%")
    
    # Next 40 (11-50)
    print("\n⭐ RECOMMENDED (11-50):")
    print("-" * 80)
    for rec in recommendations[10:50]:
        print(f"{rec['rank']:3d}. {rec['game_name']:<50s} Match: {rec['match_percentage']:5.1f}%")
    
    # Rest (51-100)
    if len(recommendations) > 50:
        print("\n💡 LEAST RECOMMENDED (51-100):")
        print("-" * 80)
        for rec in recommendations[50:100]:
            print(f"{rec['rank']:3d}. {rec['game_name']:<50s} Match: {rec['match_percentage']:5.1f}%")


def save_recommendations_to_csv(recommendations, output_file='user_recommendations.csv'):
    """Save recommendations to CSV file"""
    df = pd.DataFrame(recommendations)
    df.to_csv(output_file, index=False)
    print(f"\n✓ Recommendations saved to {output_file}")


if __name__ == "__main__":
    print("="*80)
    print("  STEAM GAME PROFILE-BASED RECOMMENDATION SYSTEM")
    print("="*80)
    
    # Example usage
    print("\nEnter your gaming preferences:")
    
    # Get user input
    preferences = input("What kind of games do you like? (or press Enter for example): ").strip()
    
    if not preferences:
        preferences = "action adventure games with great story amazing graphics multiplayer cooperative"
        print(f"Using example: {preferences}")
    
    # Optional: genres
    genres_input = input("\nEnter preferred genres (comma-separated, or press Enter to skip): ").strip()
    genres = [g.strip() for g in genres_input.split(',')] if genres_input else None
    
    # Create user profile
    user_profile = create_user_profile(preferences, genres)
    print(f"\nUser Profile: {user_profile}")
    
    # Get recommendations
    recommendations = get_top_recommendations(user_profile, top_n=100)
    
    # Display results
    display_recommendations(recommendations)
    
    # Save to CSV
    save_recommendations_to_csv(recommendations)
    
    print("\n" + "="*80)
    print("  DONE!")
    print("="*80)
