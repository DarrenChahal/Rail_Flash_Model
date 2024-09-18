# text_cleaner.py
import re
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def clean_and_lemmatize_text(texts):
    def clean_and_lemmatize_text_single(text):
        # Remove special characters and unwanted tokens (e.g., @RailMinIndia)
        text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters
        text = ' '.join(word for word in text.split() if not word.startswith('@'))  # Remove tokens starting with '@'

        # Convert to lowercase and lemmatize
        text = text.lower()
        return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return [clean_and_lemmatize_text_single(text) for text in texts]
