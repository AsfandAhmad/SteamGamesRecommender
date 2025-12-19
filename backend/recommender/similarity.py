# similarity.py
# This file calculates how similar two vectors are
# It uses Cosine Similarity to measure the angle between vectors
# The closer the angle, the more similar they are!

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def calculate_similarity(vector1, vector2):
    """
    Calculate how similar two vectors are.
    
    Args:
        vector1: First vector (e.g., user's review)
        vector2: Second vector (e.g., game review)
    
    Returns:
        Similarity score between 0 and 1
        - 1 = Exactly the same
        - 0 = Completely different
    """
    
    # Calculate cosine similarity
    similarity = cosine_similarity(vector1, vector2)[0][0]
    
    return similarity


def calculate_similarity_matrix(user_vector, all_game_vectors):
    """
    Calculate similarity between user's review and ALL games at once.
    
    Args:
        user_vector: The user's review as a vector (1 x features)
        all_game_vectors: All game reviews as vectors (n_games x features)
    
    Returns:
        Array of similarity scores, one for each game
    """
    
    # Calculate similarity with all games at once
    # This is faster than calculating one by one!
    similarities = cosine_similarity(user_vector, all_game_vectors)[0]
    
    return similarities


def find_top_matches(similarities, top_n=5):
    """
    Find the indices of the most similar games.
    
    Args:
        similarities: Array of similarity scores
        top_n: How many top matches to return (default 5)
    
    Returns:
        Indices of top N most similar games
    """
    
    # Get indices that would sort the array (highest to lowest)
    # argsort gives smallest to largest, so we reverse it with [::-1]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    return top_indices


def get_similarity_scores(similarities, indices):
    """
    Get the actual similarity scores for specific indices.
    
    Args:
        similarities: Array of all similarity scores
        indices: Which indices to get scores for
    
    Returns:
        List of similarity scores
    """
    
    scores = [similarities[i] for i in indices]
    
    return scores
