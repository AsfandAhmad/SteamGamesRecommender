# vectorizer.py
# This file converts cleaned text into numerical vectors using TF-IDF
# TF-IDF = Term Frequency - Inverse Document Frequency
# It tells us how important a word is in a document

from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_vectorizer():
    """
    Creates a TF-IDF vectorizer with simple settings.
    
    What does this do?
    - Converts text into numbers that computers can understand
    - Gives higher scores to important words
    - Gives lower scores to common words
    
    Returns:
        A TfidfVectorizer object ready to use
    """
    
    vectorizer = TfidfVectorizer(
        max_features=5000,      # Only keep 5000 most important words
        min_df=2,               # Word must appear in at least 2 documents
        max_df=0.8,             # Ignore words that appear in more than 80% of documents
        ngram_range=(1, 2)      # Use single words and two-word phrases
    )
    
    return vectorizer


def fit_vectorizer(vectorizer, documents):
    """
    Train the vectorizer on a list of documents.
    
    Args:
        vectorizer: The TfidfVectorizer object
        documents: List of text documents (cleaned reviews)
    
    Returns:
        The trained vectorizer
    """
    
    # Learn vocabulary from all documents
    vectorizer.fit(documents)
    
    print(f"✓ Vectorizer trained on {len(documents)} documents")
    print(f"✓ Vocabulary size: {len(vectorizer.vocabulary_)} words")
    
    return vectorizer


def transform_text(vectorizer, text):
    """
    Convert a single text into a numerical vector.
    
    Args:
        vectorizer: Trained TfidfVectorizer
        text: Single text document to convert
    
    Returns:
        Numerical vector representation of the text
    """
    
    # Convert text to vector
    vector = vectorizer.transform([text])
    
    return vector


def fit_and_transform(vectorizer, documents):
    """
    Train vectorizer and convert all documents to vectors in one step.
    
    Args:
        vectorizer: The TfidfVectorizer object
        documents: List of text documents
    
    Returns:
        Matrix of vectors (one vector per document)
    """
    
    # Train and convert at the same time
    vectors = vectorizer.fit_transform(documents)
    
    print(f"✓ Created vectors for {len(documents)} documents")
    print(f"✓ Vector dimensions: {vectors.shape}")
    
    return vectors
