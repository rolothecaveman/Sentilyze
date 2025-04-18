import streamlit as st
import re, string
import numpy as np
import spacy
from gensim.models.doc2vec import Doc2Vec
from tensorflow.keras.models import load_model

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Page config + CSS
st.set_page_config(page_title="main", layout="centered")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&family=Dancing+Script&display=swap');
.stApp { background-color: #051f20; }
.title-container { text-align: center; padding-top: 2rem; }
.title-container h1 {
    font-family: 'Poppins', sans-serif; font-size: 2.5rem;
    color: #f6b700; margin-bottom: 0.2rem;
}
.title-container p {
    font-family: 'Dancing Script', cursive; font-size: 1.8rem;
    color: #ffffff; margin-top: 0;
}
.stTextInput > div > input {
    background-color: #1a1a1a; color: white; border: 1px solid #ccc;
    border-right: none; border-radius: 8px 0 0 8px;
    height: 3rem; font-size: 1rem; padding: 0 0.75rem;
}
.stButton button {
    background-color: #8eb69b !important; color: white !important;
    border: none !important; border-radius: 0 8px 8px 0 !important;
    height: 3rem !important; font-size: 1rem !important;
    padding: 0 1.5rem !important; margin-top: 0.2rem !important;
    cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    d2v = Doc2Vec.load("doc2vec_model.model")  # Updated model name
    model = load_model("bilstm_model.keras")  # Updated model name
    return d2v, model

d2v_model, sentiment_model = load_models()

# Clean text
EMOJI_PATTERN = re.compile(
    "["u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    "]+", flags=re.UNICODE)

def clean_text(text):
    text = EMOJI_PATTERN.sub(r'', text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = d2v_model.infer_vector(cleaned.split()).reshape(1, -1)
    prediction = sentiment_model.predict(vector)
    return ["Positive", "Neutral", "Negative"][np.argmax(prediction)]

# Title
st.markdown("""
<div class="title-container">
    <h1>Extracting the truth from text</h1>
    <p>Sentilyze your reviews/feedback</p>
</div>
""", unsafe_allow_html=True)

# Input form
with st.form("text_form"):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_text = st.text_input("", placeholder="Paste your review here", label_visibility="collapsed")
    with col2:
        submitted = st.form_submit_button("Submit")

if submitted:
    if user_text.strip():
        sentiment = predict_sentiment(user_text)
        st.success(f"✅ Sentiment: {sentiment}")
    else:
        st.warning("⚠️ Please enter some review text.")

# Footer
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style='text-align: left;'>
        <h3 style='color: #f6b700;'>📬 Contact Us</h3>
        <p style='color: white;'>Have questions or feedback? We'd love to hear from you!</p>
        <p style='color: white;'>Email: hello@sentilyze.com</p>
        <p style='color: white;'>🌐 Website: <a href='https://sentilyze.com' style='color: #8eb69b;' target='_blank'>www.sentilyze.com</a></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='text-align: left;'>
        <h3 style='color: #f6b700;'>👥 About Us</h3>
        <p style='color: white;'>Sentilyze is a sentiment analysis tool crafted to decode the emotional tone behind feedback and reviews. We aim to empower businesses and creators to understand their audience better through clean and clear insights.</p>
    </div>
    """, unsafe_allow_html=True)
