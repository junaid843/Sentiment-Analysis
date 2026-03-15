# 🎬 IMDB Movie Review Sentiment Analyzer

A Machine Learning powered web application that analyzes movie reviews and predicts whether the sentiment is **Positive** or **Negative**.

This project uses **Natural Language Processing (NLP)** techniques and a **Logistic Regression model** trained on the **IMDB 50K Movie Reviews Dataset** to classify user input reviews.

---

## 🚀 Live Demo

Want to try the model yourself?

👉 **[Test Yourself](https://sentiment-analysis-5ei5mzyvz5igufstixuaxc.streamlit.app/)**

Click the link above and enter any movie review to see the sentiment prediction instantly.

---

## 📊 Dataset

The model is trained on the **IMDB Dataset of 50K Movie Reviews**, which contains:

* 50,000 movie reviews
* Balanced positive and negative sentiments
* Real user-generated movie feedback

Dataset Source: Kaggle

---

## ⚙️ Features

* Text preprocessing and cleaning
* Stopword removal
* Tokenization and stemming
* TF-IDF feature extraction
* Logistic Regression classification
* Interactive Streamlit web interface
* Real-time sentiment prediction

---

## 🧠 Machine Learning Pipeline

1. **Text Cleaning**

   * Lowercasing
   * Removing HTML tags
   * Removing punctuation

2. **Tokenization**

   * Splitting text into words

3. **Stopword Removal**

   * Removing common English words

4. **Stemming**

   * Reducing words to root form

5. **Vectorization**

   * TF-IDF feature extraction

6. **Model Training**

   * Logistic Regression classifier

---

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **NLTK**
* **Scikit-learn**
* **Streamlit**
* **Joblib**

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🧪 Example

Input Review:

```
This movie was absolutely fantastic! The acting and storyline were amazing.
```

Prediction:

```
Positive Sentiment
```

---

## 📌 Future Improvements

* Add Deep Learning models (LSTM / BERT)
* Add probability confidence scores
* Improve UI/UX
* Deploy using Docker

---

## 👨‍💻 Author

Developed by **Junaid Ahmed**

If you found this project useful, consider giving it a ⭐ on GitHub.
