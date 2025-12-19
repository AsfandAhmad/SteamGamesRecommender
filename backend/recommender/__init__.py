# __init__.py
# This file makes the 'recommender' folder a Python package
# It also makes importing easier by exposing the main classes and functions

from recommender.preprocess import clean_text
from recommender.vectorizer import (
    create_tfidf_vectorizer,
    fit_vectorizer,
    transform_text,
    fit_and_transform
)
from recommender.similarity import (
    calculate_similarity,
    calculate_similarity_matrix,
    find_top_matches,
    get_similarity_scores
)
from recommender.recommend import GameRecommender

# This allows users to do:
# from recommender import GameRecommender, clean_text
# Instead of:
# from recommender.recommend import GameRecommender
# from recommender.preprocess import clean_text

__all__ = [
    'clean_text',
    'create_tfidf_vectorizer',
    'fit_vectorizer',
    'transform_text',
    'fit_and_transform',
    'calculate_similarity',
    'calculate_similarity_matrix',
    'find_top_matches',
    'get_similarity_scores',
    'GameRecommender'
]
