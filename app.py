

# Import necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import joblib

# Streamlit Page Configuration
st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬", layout="centered")

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Initialize Stemmer and Stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# 1. TEXT PREPROCESSING
def preprocess_text(text):
    """
    Clean and preprocess text data
    """
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 2. TOKENIZATION AND STEMMING
def tokenize_and_stem(text):
    """
    Tokenize text, remove stopwords, and apply stemming
    """
    tokens = nltk.word_tokenize(text)
    stemmed_tokens =[stemmer.stem(token) for token in tokens
                     if token not in stop_words and len(token) > 2]
    return ' '.join(stemmed_tokens)

# Cache the dataset loading and model training so it runs ONLY ONCE
@st.cache_resource
def load_and_train_model():
    st.info("Training the model for the first time... Please wait.")
    
    # Load the dataset
    #df = pd.read_csv('IMDB Dataset.csv', on_bad_lines='skip', engine='python')
    # Download dataset from Kaggle
    path = kagglehub.dataset_download(
        "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
    )

    csv_path = os.path.join(path, "IMDB Dataset.csv")

# Load the dataset
    df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')
    # Apply preprocessing
    df['processed_review'] = df['review'].apply(preprocess_text)
    df['processed_review'] = df['processed_review'].apply(tokenize_and_stem)

    # Prepare features and labels
    X = df['processed_review']
    y = df['sentiment'].map({'positive': 1, 'negative': 0})

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. TEXT VECTORIZATION
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8
    )

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # 4. TRAIN LOGISTIC REGRESSION MODEL
    logreg_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )

    logreg_model.fit(X_train_tfidf, y_train)

    # Save the model and vectorizer for later use
    joblib.dump(logreg_model, 'sentiment_model.pkl')
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
    
    return logreg_model, tfidf_vectorizer

# Call the cached function to get model and vectorizer
logreg_model, tfidf_vectorizer = load_and_train_model()

# Prediction function
def predict_sentiment(review):
    """
    Predict sentiment for a new review
    """
    processed = preprocess_text(review)
    processed = tokenize_and_stem(processed)
    vectorized = tfidf_vectorizer.transform([processed])

    prediction = logreg_model.predict(vectorized)[0]
    probability = logreg_model.predict_proba(vectorized)[0]

    confidence = max(probability) * 100

    if prediction == 1:
        sentiment = "Positive 😊"
    else:
        sentiment = "Negative 😞"

    return sentiment, f"{confidence:.2f}%"

# ---------------------------------------
# 5. STREAMLIT UI (PROFESSIONAL UI)
# ---------------------------------------

st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.markdown("### Logistic Regression NLP Model")
st.write("This application analyzes movie reviews and predicts whether the sentiment is **Positive** or **Negative**.")

# Layout with columns
col1, col2 = st.columns([1.5, 1])

with col1:
    review_input = st.text_area(
        "Movie Review", 
        height=180, 
        placeholder="Type your movie review here..."
    )
    analyze_btn = st.button("Analyze Sentiment 🔍", type="primary")

with col2:
    st.markdown("<br>", unsafe_allow_html=True) # Just for spacing alignment
    st.markdown("### Results")
    
    if analyze_btn:
        if review_input.strip() == "":
            st.warning("Please enter a review first!")
        else:
            with st.spinner("Analyzing..."):
                sentiment, confidence = predict_sentiment(review_input)
                st.success("Analysis Complete!")
                st.markdown(f"**Predicted Sentiment:** {sentiment}")
                st.markdown(f"**Confidence Score:** {confidence}")
    else:
        st.info("Enter a review and click analyze to see the results here.")

st.markdown("---")
st.markdown("### 🧪 Example Reviews")
st.code("This movie was absolutely fantastic! Great acting and storyline.", language=None)
st.code("Terrible film, waste of time. Poor acting and boring plot.", language=None)
st.code("An okay movie with some good moments but overall average.", language=None)

st.markdown("---")
st.markdown(
    """
    🔧 **Model:** Logistic Regression | 
    📊 **Dataset:** IMDB 50K Movie Reviews | 
    🧠 **Vectorization:** TF-IDF
    """
)
