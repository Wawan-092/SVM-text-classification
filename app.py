import streamlit as st
import joblib
import ssl
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
# Preprocessing
def preprocess_text(text):
    # Tokenisasi
    tokens = nltk.word_tokenize(text)

    # Konversi ke huruf kecil
    tokens = [token.lower() for token in tokens]

    # Penghapusan kata-kata stopword
    stop_words = set(stopwords.words('indonesian'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens

# Load the trained model and vectorizer
model = joblib.load("model.pkl")
tfidf_vect_8020 = joblib.load("vectorizer.pkl")

def classify_review(review):
    # Preprocess the input text
    review = " ".join(preprocess_text(review))
    review = tfidf_vect_8020.transform([review])
    # Make predictions
    prediction = model.predict(review)
    return prediction[0]

# Create a Streamlit web app
def main():
    nltk.download('punkt')
    st.title("CEK ULASAN DONG")
    st.write("Masukkan sebuah ulasan dan klik tombol 'OKE' untuk menggolongkan jenis ulasan (postif atau negatif).")
    
    
    # Input text box
    review = st.text_area("Masukkan Ulasan Anda", "")
    
    # Classify button
    if st.button("OKE"):
        if review:
            # Perform classification
            prediction = "Positif" if classify_review(review) == 1 else "Negatif"
            st.write("Termasuk Jenis Ulasan:", prediction)
        else:
            st.write("Silakan masukkan sebuah ulasan.")

if __name__ == "__main__":
    main()
