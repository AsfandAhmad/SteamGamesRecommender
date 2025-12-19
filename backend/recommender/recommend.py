# recommend.py
# This file brings everything together to make game recommendations
# It uses preprocess, vectorizer, and similarity to recommend games

from recommender.preprocess import clean_text
from recommender.vectorizer import transform_text
from recommender.similarity import calculate_similarity_matrix, find_top_matches, get_similarity_scores


class GameRecommender:
    """
    Main recommendation system class.
    This combines all the pieces to recommend games based on user reviews.
    """
    
    def __init__(self, vectorizer, game_vectors, game_names, game_ids):
        """
        Initialize the recommender with trained data.
        
        Args:
            vectorizer: Trained TF-IDF vectorizer
            game_vectors: Pre-computed vectors for all games
            game_names: List of game names
            game_ids: List of game IDs
        """
        self.vectorizer = vectorizer
        self.game_vectors = game_vectors
        self.game_names = game_names
        self.game_ids = game_ids
        
    def recommend(self, user_review, top_n=5):
        """
        Get game recommendations based on user's review text.
        
        This is the main function that does everything:
        1. Clean the user's review
        2. Convert it to a vector
        3. Find similar games
        4. Return top recommendations
        
        Args:
            user_review: Raw text from user describing what they want
            top_n: Number of recommendations to return (default 5)
        
        Returns:
            List of dictionaries with game recommendations
        """
        
        # Step 1: Clean the user's review text
        cleaned_review = clean_text(user_review)
        
        if not cleaned_review.strip():
            # If cleaning removed everything, return empty list
            return []
        
        # Step 2: Convert cleaned review to vector
        user_vector = transform_text(self.vectorizer, cleaned_review)
        
        # Step 3: Calculate similarity with all games
        similarities = calculate_similarity_matrix(user_vector, self.game_vectors)
        
        # Step 4: Find top N most similar games
        top_indices = find_top_matches(similarities, top_n=top_n)
        top_scores = get_similarity_scores(similarities, top_indices)
        
        # Step 5: Format results as a list of dictionaries
        recommendations = []
        for idx, score in zip(top_indices, top_scores):
            recommendations.append({
                'game_id': self.game_ids[idx],
                'game_name': self.game_names[idx],
                'similarity_score': float(score),
                'match_percentage': float(score * 100)
            })
        
        return recommendations
    
    def get_all_games(self):
        """
        Get a list of all available games.
        
        Returns:
            List of dictionaries with game info
        """
        games = []
        for game_id, game_name in zip(self.game_ids, self.game_names):
            games.append({
                'game_id': game_id,
                'game_name': game_name
            })
        return games
    
    def get_total_games(self):
        """
        Get the total number of games in the system.
        
        Returns:
            Integer count of games
        """
        return len(self.game_names)
