import re 
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    filtered_words = []
    for w in words:
     if w not in stop_words:
        filtered_words.append(w)
    words = filtered_words
    return " ".join(words)