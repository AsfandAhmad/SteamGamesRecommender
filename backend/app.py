# app.py
# Flask API for Steam Game Recommendation System

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import pandas as pd
from recommender import GameRecommender
from config import (
    VECTORIZER_PATH,
    GAME_VECTORS_PATH,
    PROCESSED_GAMES_FILE,
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG,
    CORS_ORIGINS,
    DEFAULT_TOP_N,
    MAX_TOP_N
)

app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS)

# Global recommender instance
recommender = None


def load_recommender():
    """Load the trained model and initialize recommender"""
    global recommender
    
    try:
        # Check if model files exist
        if not os.path.exists(VECTORIZER_PATH):
            print(f"Error: Vectorizer not found at {VECTORIZER_PATH}")
            return False
        
        if not os.path.exists(GAME_VECTORS_PATH):
            print(f"Error: Game vectors not found at {GAME_VECTORS_PATH}")
            return False
        
        if not os.path.exists(PROCESSED_GAMES_FILE):
            print(f"Error: Processed games file not found at {PROCESSED_GAMES_FILE}")
            return False
        
        # Load vectorizer
        print("Loading vectorizer...")
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load game vectors
        print("Loading game vectors...")
        with open(GAME_VECTORS_PATH, 'rb') as f:
            game_vectors = pickle.load(f)
        
        # Load game data
        print("Loading game data...")
        games_df = pd.read_csv(PROCESSED_GAMES_FILE)
        game_names = games_df['app_name'].tolist()
        game_ids = games_df['app_id'].tolist()
        
        # Initialize recommender
        recommender = GameRecommender(vectorizer, game_vectors, game_names, game_ids)
        
        print(f"✓ Recommender loaded successfully with {recommender.get_total_games()} games")
        return True
        
    except Exception as e:
        print(f"Error loading recommender: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'Steam Game Recommendation API',
        'model_loaded': recommender is not None,
        'total_games': recommender.get_total_games() if recommender else 0
    })


@app.route('/api/games', methods=['GET'])
def get_games():
    """Get list of all available games"""
    if recommender is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded',
            'message': 'Recommendation system is not initialized'
        }), 503
    
    try:
        games = recommender.get_all_games()
        
        # Optional pagination
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        start = (page - 1) * per_page
        end = start + per_page
        
        return jsonify({
            'success': True,
            'total_games': len(games),
            'page': page,
            'per_page': per_page,
            'games': games[start:end]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Get game recommendations based on user review"""
    if recommender is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded',
            'message': 'Recommendation system is not initialized'
        }), 503
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'review' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing review',
                'message': 'Please provide a review text in the request body'
            }), 400
        
        user_review = data['review']
        top_n = data.get('top_n', DEFAULT_TOP_N)
        
        # Validate inputs
        if not isinstance(user_review, str) or len(user_review.strip()) == 0:
            return jsonify({
                'success': False,
                'error': 'Invalid review',
                'message': 'Review text must be a non-empty string'
            }), 400
        
        # Limit top_n
        if not isinstance(top_n, int) or top_n < 1:
            top_n = DEFAULT_TOP_N
        if top_n > MAX_TOP_N:
            top_n = MAX_TOP_N
        
        # Get recommendations
        recommendations = recommender.recommend(user_review, top_n=top_n)
        
        if not recommendations:
            return jsonify({
                'success': True,
                'user_review': user_review,
                'recommendations': [],
                'message': 'No matching games found. Try using different keywords.'
            })
        
        return jsonify({
            'success': True,
            'user_review': user_review,
            'total_recommendations': len(recommendations),
            'recommendations': recommendations
        })
        
    except Exception as e:
        print(f"Error in recommend endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred while processing your request'
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about the recommendation system"""
    if recommender is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 503
    
    try:
        return jsonify({
            'success': True,
            'stats': {
                'total_games': recommender.get_total_games(),
                'vocabulary_size': len(recommender.vectorizer.vocabulary_),
                'vector_dimensions': recommender.game_vectors.shape[1]
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Not found',
        'message': 'The requested endpoint does not exist'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Steam Game Recommendation API Server")
    print("="*60 + "\n")
    
    # Load the model
    print("Loading recommendation model...")
    if load_recommender():
        print("\n" + "="*60)
        print("  Server is ready!")
        print(f"  API available at: http://{FLASK_HOST}:{FLASK_PORT}")
        print("  Endpoints:")
        print("    GET  /              - Health check")
        print("    GET  /api/games     - List all games")
        print("    POST /api/recommend - Get recommendations")
        print("    GET  /api/stats     - System statistics")
        print("="*60 + "\n")
        
        # Run the Flask app
        app.run(
            host=FLASK_HOST,
            port=FLASK_PORT,
            debug=FLASK_DEBUG
        )
    else:
        print("\n" + "="*60)
        print("  ERROR: Failed to load model")
        print("  Please train the model first by running:")
        print("  python train_model.py")
        print("="*60 + "\n")
