import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Apply custom CSS for Twitter-themed background with white Twitter bird logos
st.markdown(
    """
    <style>
        /* Set the entire page background to Twitter sky blue */
        [data-testid="stAppViewContainer"] {
            background-color: #1DA1F2;
            background-image: url('https://upload.wikimedia.org/wikipedia/sco/9/9f/Twitter_bird_logo_2012.svg');
            background-size: 50px;
            background-repeat: repeat;
            background-position: top left;
        }

        /* Style the text input box (black background, white text) */
        textarea {
            background-color: black !important;
            color: white !important;
        }

        /* Customize Streamlit's main content area */
        [data-testid="stVerticalBlock"] {
            background-color: transparent !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Download stopwords once, using Streamlit's caching
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# Load model and vectorizer once
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Define sentiment prediction function
def predict_sentiment(text, model, vectorizer, stop_words):
    # Preprocess text
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text = [text]
    text = vectorizer.transform(text)

    # Predict sentiment
    sentiment = model.predict(text)
    return "Negative" if sentiment == 0 else "Positive"

# Function to create a colored sentiment card
def create_card(text, sentiment):
    color = "green" if sentiment == "Positive" else "red"
    st.markdown(f"""
        <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
            <h5 style="color: white; text-align: center;">{sentiment} Sentiment</h5>
            <p style="color: white; text-align: center;">{text}</p>
        </div>
    """, unsafe_allow_html=True)

# Main app logic
def main():
    st.title("Twitter Sentiment Analysis")

    # Load stopwords, model, and vectorizer
    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    text_input = st.text_area("Enter text for analysis")
    if st.button("Analyze"):
        if text_input.strip():
            sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
            create_card(text_input, sentiment)
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
