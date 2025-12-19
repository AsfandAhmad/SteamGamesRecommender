# streamlit_app.py
# Streamlit frontend for Steam Game Recommender

import streamlit as st
import pandas as pd
import pickle
import sys
import os

# Download NLTK data before imports (for Streamlit Cloud)
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# Add backend to path
current_dir = os.getcwd()
backend_path = os.path.join(current_dir, 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from recommender import GameRecommender
from config import VECTORIZER_PATH, GAME_VECTORS_PATH, PROCESSED_GAMES_FILE


@st.cache_resource
def load_recommender():
    """Load the recommendation model (cached)"""
    try:
        # Ensure model directory exists
        model_dir = os.path.dirname(VECTORIZER_PATH)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        
        # Check if model files exist
        if not os.path.exists(VECTORIZER_PATH):
            st.error(f"❌ Model file not found: {VECTORIZER_PATH}")
            st.info("📋 Files in backend/models: " + str(os.listdir(model_dir) if os.path.exists(model_dir) else "Directory doesn't exist"))
            return None, None
            
        if not os.path.exists(GAME_VECTORS_PATH):
            st.error(f"❌ Model file not found: {GAME_VECTORS_PATH}")
            return None, None
        
        # Load vectorizer
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load game vectors
        with open(GAME_VECTORS_PATH, 'rb') as f:
            game_vectors = pickle.load(f)
        
        # Load game data
        games_df = pd.read_csv(PROCESSED_GAMES_FILE)
        game_names = games_df['app_name'].tolist()
        game_ids = games_df['app_id'].tolist()
        
        # Create recommender
        recommender = GameRecommender(vectorizer, game_vectors, game_names, game_ids)
        
        return recommender, games_df
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def main():
    # Page config
    st.set_page_config(
        page_title="Steam Game Recommender",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS - Steam Gaming Theme
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #1b2838 0%, #2a475e 100%);
        }
        
        .main {
            padding: 2rem;
        }
        
        .stButton>button {
            background: linear-gradient(90deg, #06BFFF 0%, #2D73FF 100%);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(6, 191, 255, 0.3);
        }
        
        .stButton>button:hover {
            background: linear-gradient(90deg, #2D73FF 0%, #06BFFF 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(6, 191, 255, 0.5);
        }
        
        .stTextArea textarea, .stTextInput input {
            background: rgba(27, 40, 56, 0.9);
            border: 2px solid #66c0f4;
            color: white;
            border-radius: 5px;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            background: linear-gradient(135deg, rgba(27, 40, 56, 0.95) 0%, rgba(42, 71, 94, 0.95) 100%);
            padding: 1.5rem 2rem;
            border-radius: 15px;
            border: 2px solid #66c0f4;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: rgba(27, 40, 56, 0.5);
            color: #66c0f4;
            font-weight: 700;
            font-size: 1.2rem;
            border-radius: 10px;
            padding: 1rem 2rem;
            border: 2px solid transparent;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(102, 192, 244, 0.2);
            border-color: #66c0f4;
            transform: translateY(-2px);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #06BFFF 0%, #2D73FF 100%);
            color: white;
            border-color: #1A9FFF;
            box-shadow: 0 5px 20px rgba(6, 191, 255, 0.5);
            transform: translateY(-3px);
        }
        
        h1, h2, h3, h4 {
            color: #66c0f4;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }
        
        .game-card {
            background: linear-gradient(135deg, rgba(27, 40, 56, 0.9) 0%, rgba(42, 71, 94, 0.9) 100%);
            border: 2px solid #66c0f4;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 5px 15px rgba(102, 192, 244, 0.2);
            transition: all 0.3s ease;
        }
        
        .game-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(102, 192, 244, 0.4);
        }
        
        p, label, .stMarkdown {
            color: #c7d5e0;
        }
        
        .tab-content {
            background: rgba(27, 40, 56, 0.7);
            padding: 2rem;
            border-radius: 15px;
            border: 2px solid rgba(102, 192, 244, 0.3);
            margin-top: 1.5rem;
        }
        
        .section-header {
            background: linear-gradient(90deg, #1b2838 0%, #66c0f4 100%);
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 1.5rem;
            border: 2px solid #66c0f4;
            box-shadow: 0 5px 20px rgba(102, 192, 244, 0.3);
        }
        
        .stCheckbox {
            background: rgba(27, 40, 56, 0.6);
            padding: 0.5rem;
            border-radius: 5px;
            border: 1px solid #66c0f4;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1b2838 0%, #1A9FFF 100%); 
                padding: 2rem; border-radius: 10px; text-align: center; margin-bottom: 2rem; 
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);'>
        <h1 style='color: white; font-size: 3rem; margin: 0;'>🎮 STEAM GAME RECOMMENDER</h1>
        <p style='color: #66c0f4; font-size: 1.3rem; margin-top: 1rem;'>Built & Powered by Asfand Ahmed 23k-0698</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn.cloudflare.steamstatic.com/store/home/store_nav_bg.png", use_container_width=True)
        st.markdown("### ⚙️ Settings")
        top_n = st.slider("Number of recommendations", min_value=5, max_value=100, value=10, step=5)
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.info("""
        This recommendation system uses:
        - **TF-IDF Vectorization**
        - **Cosine Similarity**
        - Natural Language Processing
        
        Trained on thousands of Steam reviews!
        """)
        
        st.markdown("---")
        st.markdown("### 🔗 Quick Links")
        st.markdown("- [Steam Store](https://store.steampowered.com)")
        st.markdown("- [GitHub](https://github.com)")
    
    # Load model
    recommender, games_df = load_recommender()
    
    if recommender is None:
        st.error("❌ Failed to load recommendation model. Please run `python backend/train_model.py` first.")
        st.info("💡 Run: `cd backend && python train_model.py`")
        return
    
    # Success message with metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎮 Total Games", recommender.get_total_games())
    with col2:
        st.metric("📚 Vocabulary Size", f"{len(recommender.vectorizer.vocabulary_):,}")
    with col3:
        st.metric("🔢 Vector Dimensions", recommender.game_vectors.shape[1])
    
    st.markdown("---")
    
    # Main content - Two tabs
    tab1, tab2 = st.tabs(["🔍 **Search by Preferences**", "📊 **Top 100 Rankings**"])
    
    # Tab 1: Search by user review
    with tab1:
        st.markdown("""
        <div class='section-header'>
            <h2 style='color: white; margin: 0;'>🎯 TELL US WHAT YOU'RE LOOKING FOR</h2>
            <p style='color: #c7d5e0; margin-top: 0.5rem; font-size: 1.1rem;'>Describe your ideal game and we'll find the perfect matches</p>
        </div>
        """, unsafe_allow_html=True)
        # Text input with better styling
        user_review = st.text_area(
            "✍️ Your gaming preferences:",
            placeholder="Example: I love action-packed shooters with stunning graphics, intense multiplayer battles, and strategic gameplay...",
            height=120,
            help="Be specific! Mention genres, gameplay styles, graphics quality, or any features you enjoy."
        )
        
        # Example buttons with better layout
        st.markdown("""
        <div style='background: rgba(27, 40, 56, 0.8); padding: 1rem; border-radius: 10px; 
                    border: 2px solid #66c0f4; margin: 1.5rem 0;'>
            <h3 style='color: #66c0f4; margin: 0; text-align: center;'>💡 QUICK EXAMPLES</h3>
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("⚔️ Action/Combat", use_container_width=True):
                user_review = "fast-paced action shooter with intense combat and great graphics"
                
        with col2:
            if st.button("🧩 Puzzle/Strategy", use_container_width=True):
                user_review = "challenging puzzle games with creative solutions and relaxing music"
                
        with col3:
            if st.button("👥 Multiplayer", use_container_width=True):
                user_review = "fun cooperative multiplayer games to play with friends online"
                
        with col4:
            if st.button("📖 Story/RPG", use_container_width=True):
                user_review = "immersive RPG with deep story amazing characters and exploration"
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Search button
        search_col1, search_col2, search_col3 = st.columns([1, 2, 1])
        with search_col2:
            search_button = st.button("🚀 Find My Perfect Games", type="primary", use_container_width=True)
        
        if search_button:
            if user_review.strip():
                with st.spinner("🔍 Analyzing your preferences..."):
                    recommendations = recommender.recommend(user_review, top_n=top_n)
                
                if recommendations:
                    st.success(f"✨ Found {len(recommendations)} amazing games for you!")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Display recommendations with better styling
                    for i, rec in enumerate(recommendations, 1):
                        match_pct = rec['match_percentage']
                        
                        # Color coding
                        if match_pct >= 70:
                            border_color = "#4CAF50"
                            icon = "🔥"
                            badge = "Excellent Match"
                        elif match_pct >= 40:
                            border_color = "#2196F3"
                            icon = "⭐"
                            badge = "Good Match"
                        else:
                            border_color = "#FF9800"
                            icon = "💡"
                            badge = "Fair Match"
                        
                        st.markdown(f"""
                        <div class='game-card' style="border-left: 5px solid {border_color};">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <h3 style="margin: 0; color: #66c0f4;">{icon} #{i} - {rec['game_name']}</h3>
                                    <p style="margin: 0.5rem 0; color: #c7d5e0;">🆔 {rec['game_id']}</p>
                                </div>
                                <div style="text-align: right;">
                                    <h2 style="margin: 0; color: {border_color}; text-shadow: 0 0 10px {border_color};">{match_pct:.1f}%</h2>
                                    <p style="margin: 0; color: #66c0f4; font-weight: bold;">{badge}</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Download as CSV
                    st.markdown("<br>", unsafe_allow_html=True)
                    df_results = pd.DataFrame(recommendations)
                    csv = df_results.to_csv(index=False)
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.download_button(
                            label="📥 DOWNLOAD RESULTS AS CSV",
                            data=csv,
                            file_name="recommendations.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
                else:
                    st.warning("No recommendations found. Try different keywords!")
            
            else:
                st.error("Please enter your gaming preferences!")
                st.error("Please enter your gaming preferences!")
    
    # Tab 2: Top 100 recommendations
    with tab2:
        st.markdown("""
        <div class='section-header'>
            <h2 style='color: white; margin: 0;'>🏆 TOP 100 GAME RANKINGS</h2>
            <p style='color: #c7d5e0; margin-top: 0.5rem; font-size: 1.1rem;'>Create your gaming profile and get personalized rankings</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Profile input
        profile_text = st.text_area(
            "✍️ Your gaming profile:",
            placeholder="Describe your ideal gaming experience: graphics, gameplay, multiplayer, story, genres...",
            height=100,
            key="profile",
            help="The more specific you are, the better recommendations you'll get!"
        )
        
        # Genre tags
        st.markdown("""
        <div style='background: rgba(27, 40, 56, 0.8); padding: 1rem; border-radius: 10px; 
                    border: 2px solid #66c0f4; margin: 1.5rem 0;'>
            <h3 style='color: #66c0f4; margin: 0 0 1rem 0; text-align: center;'>🎮 SELECT YOUR FAVORITE GENRES</h3>
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        genres = []
        with col1:
            if st.checkbox("Action"): genres.append("action")
            if st.checkbox("Adventure"): genres.append("adventure")
            if st.checkbox("RPG"): genres.append("rpg")
        
        with col2:
            if st.checkbox("Strategy"): genres.append("strategy")
            if st.checkbox("Simulation"): genres.append("simulation")
            if st.checkbox("Sports"): genres.append("sports")
        
        with col3:
            if st.checkbox("Racing"): genres.append("racing")
            if st.checkbox("Puzzle"): genres.append("puzzle")
            if st.checkbox("Horror"): genres.append("horror")
        
        with col4:
            if st.checkbox("Multiplayer"): genres.append("multiplayer")
            if st.checkbox("Indie"): genres.append("indie")
            if st.checkbox("Casual"): genres.append("casual")
        
        # Generate button
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_btn = st.button("📊 GENERATE TOP 100 RANKINGS", type="primary", key="generate_top100", use_container_width=True)
        
        if generate_btn:
            # Combine profile with genres
            full_profile = profile_text
            if genres:
                full_profile = f"{profile_text} {' '.join(genres)}"
            
            if full_profile.strip():
                with st.spinner("Generating top 100 recommendations..."):
                    recommendations = recommender.recommend(full_profile, top_n=100)
                
                if recommendations:
                    st.success("✅ Generated top 100 recommendations!")
                    
                    # Add categories
                    for rec in recommendations:
                        rank = recommendations.index(rec) + 1
                        rec['rank'] = rank
                        if rank <= 10:
                            rec['category'] = '🔥 Highly Recommended'
                        elif rank <= 50:
                            rec['category'] = '⭐ Recommended'
                        else:
                            rec['category'] = '💡 Least Recommended'
                    
                    # Display in sections with better styling
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 🔥 Top 10 - Highly Recommended")
                    
                    for i, rec in enumerate(recommendations[:10], 1):
                        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
                        st.markdown(f"""
                        <div style="padding: 1rem; background: linear-gradient(90deg, #ff6b6b 0%, #ee5a6f 100%); 
                                    border-radius: 10px; margin-bottom: 0.5rem; color: white;">
                            <h4 style="margin: 0;">{medal} {rec['game_name']} - {rec['match_percentage']:.1f}%</h4>
                            <p style="margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.9rem;">Game ID: {rec['game_id']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### ⭐ Recommended (11-50)")
                    df_mid = pd.DataFrame(recommendations[10:50])
                    df_mid = df_mid[['rank', 'game_name', 'game_id', 'match_percentage']]
                    df_mid.columns = ['Rank', 'Game Name', 'Game ID', 'Match %']
                    st.dataframe(df_mid, use_container_width=True, height=400)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 💡 Explore More (51-100)")
                    df_low = pd.DataFrame(recommendations[50:100])
                    df_low = df_low[['rank', 'game_name', 'game_id', 'match_percentage']]
                    df_low.columns = ['Rank', 'Game Name', 'Game ID', 'Match %']
                    st.dataframe(df_low, use_container_width=True, height=400)
                    
                    # Download full list
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    df_full = pd.DataFrame(recommendations)
                    csv = df_full.to_csv(index=False)
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.download_button(
                            label="📥 DOWNLOAD FULL TOP 100 AS CSV",
                            data=csv,
                            file_name="top_100_recommendations.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
            
            else:
                st.error("Please enter your gaming profile or select genres!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p style="margin: 0;">🎮 <strong>Steam Game Recommendation System</strong> 🎮</p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            Powered by <strong>TF-IDF Vectorization</strong> & <strong>Cosine Similarity</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
