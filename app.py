import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ===============================
# REQUIRED FUNCTION (FOR PICKLE)
# MUST MATCH TRAINING FUNCTION NAME
# ===============================
def clean_text_series(texts):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    cleaned_list = []
    for text in texts:
        text = re.sub(r"[^a-zA-Z\s]", "", str(text))
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [w for w in tokens if w not in stop_words]
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
        cleaned_list.append(" ".join(tokens))

    return cleaned_list

# ===============================
# NLTK DOWNLOAD
# ===============================
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ===============================
# UI
# ===============================
st.title("📦 Flipkart Sentiment Analyzer")

review = st.text_area("Enter Review Text")

if st.button("Predict"):

    if review.strip() == "":
        st.warning("Enter some text")
    else:
        prediction = model.predict([review])[0]

        if prediction == 1:
            st.success("Positive Sentiment")
        else:
            st.error("Negative Sentiment")

