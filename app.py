
import streamlit as st
import pickle
import re
import nltk
import matplotlib.pyplot as plt
from scipy.sparse import hstack

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

st.set_page_config(page_title="Hate Speech Detector", page_icon="🛡️", layout="centered")

@st.cache_resource
def load_model():
    with open('hate_speech_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_word = pickle.load(f)
    with open('tfidf_char.pkl', 'rb') as f:
        tfidf_char = pickle.load(f)
    return model, tfidf_word, tfidf_char

model, tfidf_word, tfidf_char = load_model()
stop_words = set(stopwords.words('english'))

def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'rt\s+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(w for w in text.split() if w not in stop_words)
    return text.strip()

def predict(text):
    cleaned = clean_tweet(text)
    w = tfidf_word.transform([cleaned])
    c = tfidf_char.transform([cleaned])
    features = hstack([w, c])
    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    return pred, probs

st.title("🛡️ Hate Speech Detector")
st.markdown("Detects **Hate Speech**, **Offensive Language**, or **Neither** in tweets.")
st.markdown("---")

tweet_input = st.text_area("✍️ Enter a tweet:", height=120, placeholder="Type something here...")

if st.button("🔍 Analyze", use_container_width=True):
    if tweet_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction, probs = predict(tweet_input)
        labels = {0: "🚨 Hate Speech", 1: "⚠️ Offensive Language", 2: "✅ Neither"}
        descriptions = {
            0: "This tweet contains **hate speech**.",
            1: "This tweet contains **offensive language**.",
            2: "This tweet is **safe**."
        }
        st.markdown(f"### Result: {labels[prediction]}")
        st.markdown(descriptions[prediction])
        st.markdown("#### Confidence Scores")
        fig, ax = plt.subplots(figsize=(6, 2.5))
        bar_labels = ['Hate Speech', 'Offensive', 'Neither']
        bar_colors = ['#e74c3c', '#e67e22', '#2ecc71']
        bars = ax.barh(bar_labels, probs, color=bar_colors)
        ax.set_xlim(0, 1)
        for bar, val in zip(bars, probs):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.1%}', va='center')
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
