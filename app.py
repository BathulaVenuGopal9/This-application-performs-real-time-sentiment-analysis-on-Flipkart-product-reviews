import streamlit as st

# ===============================
# PAGE CONFIG MUST BE FIRST
# ===============================
st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    page_icon="📦",
    layout="centered"
)

import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ===============================
# DOWNLOAD NLTK (SAFE FOR CLOUD)
# ===============================
@st.cache_resource
def download_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

download_nltk()

# ===============================
# TEXT CLEANING (MATCH TRAINING)
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
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    try:
        with open("best_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ===============================
# UI
# ===============================
st.title("📦 Flipkart Sentiment Analyzer")
st.write("Enter a product review to predict sentiment.")

review = st.text_area("✍️ Enter Review Text")

# ===============================
# PREDICTION
# ===============================
if st.button("Predict"):

    if model is None:
        st.error("Model not loaded. Check best_model.pkl")

    elif review.strip() == "":
        st.warning("Please enter some text.")

    else:
        try:
            prediction = model.predict([review])[0]

            confidence = None
            if hasattr(model, "predict_proba"):
                confidence = model.predict_proba([review]).max()

            if prediction == 1:
                st.success("😊 Positive Sentiment")
            else:
                st.error("😞 Negative Sentiment")

            if confidence is not None:
                st.info(f"Confidence: {confidence:.2f}")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | ML Sentiment Model")


