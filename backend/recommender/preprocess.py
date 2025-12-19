# preprocess.py
# This file contains all text preprocessing logic
# It cleans raw review text so it can be used for TF-IDF

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources (only runs first time)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Initialize lemmatizer and stopword list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    """
    This function takes raw review text and returns cleaned text.
    Steps:
    1. Convert text to lowercase
    2. Remove punctuation and numbers
    3. Remove stopwords
    4. Apply lemmatization
    """
    
    # Handle NaN or empty values
    if not isinstance(text, str) or not text:
        return ""

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove punctuation and numbers using regex
    text = re.sub(r'[^a-z\s]', '', text)

    # 3. Split text into individual words
    words = text.split()

    # 4. Remove stopwords and apply lemmatization
    cleaned_words = []
    for word in words:
        if word not in stop_words:
            # Convert word to its base form
            lemma = lemmatizer.lemmatize(word)
            cleaned_words.append(lemma)

    # 5. Join words back into a single string
    cleaned_text = " ".join(cleaned_words)

    return cleaned_text
