'''The Capstone Project onjectives - To design a Python program to perform sentiment analysis on the dataset of product reviews using spaCy and TextBlob by:
    Loading the spaCy model in English.
    Preprocessing the text data by removing stopwords and performing basic text cleaning.
    Creating a function for sentiment analysis that uses the spaCy model and TextBlob to determine sentiment polarity.
    Testing the sentiment analysis function on sample product reviews and comparing the similarity of two product reviews.'''
    
# Import appropriate the spaCy model
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Add the SpacyTextBlob component to the spaCy pipeline
nlp.add_pipe('spacytextblob')

# Load the data from the csv file into a pandas dataframe to allow data inspection and preprocessing
df = pd.read_csv(r'C:\Users\User\Dropbox\KP23100009714\Data Science (Fundamentals)\T21 - Capstone Project - NLP Applications\1429_1.csv')

# Preprocess the text data
# Remove missing values from the 'reviews.text' column
clean_data = df.dropna(subset=['reviews.text'])

# Define a function for sentiment analysis
def analyze_sentiment(review_text):
    doc = nlp(review_text)
    # Remove stopwords and perform basic text cleaning
    tokens = [token.text.lower().strip() for token in doc if not token.is_stop and token.text.strip() != '']
    clean_review = ' '.join(tokens)
    # Analyze sentiment
    doc = nlp(clean_review)
    polarity = doc._.blob.polarity
    sentiment = 'positive' if polarity > 0 else 'negative' if polarity < 0 else 'neutral'
    return sentiment, polarity

# Test the sentiment analysis function on sample product reviews
sample_reviews = clean_data['reviews.text'][:5]  # Taking the first 5 reviews for testing
for review in sample_reviews:
    sentiment, polarity = analyze_sentiment(review)
    print(f"Review: {review}\nSentiment: {sentiment}, Polarity: {polarity}\n")

# Compare the similarity of two product reviews
review1 = clean_data['reviews.text'][0]
review2 = clean_data['reviews.text'][1]
doc1 = nlp(review1)
doc2 = nlp(review2)
similarity_score = doc1.similarity(doc2)
print(f"Similarity between two reviews: {similarity_score}")