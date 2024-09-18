# from flask import Flask, request, jsonify
# import joblib
# import re
# from nltk.stem import WordNetLemmatizer

# # Initialize Flask app
# app = Flask(__name__)

# # Initialize lemmatizer
# lemmatizer = WordNetLemmatizer()

# def clean_and_lemmatize_text(texts):
#     def clean_and_lemmatize_text_single(text):
#         # Remove special characters and unwanted tokens (e.g., @RailMinIndia)
#         text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters
#         text = ' '.join(word for word in text.split() if not word.startswith('@'))  # Remove tokens starting with '@'
        
#         # Convert to lowercase and lemmatize
#         text = text.lower()
#         return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

#     return [clean_and_lemmatize_text_single(text) for text in texts]

# # Load the model
# model = joblib.load('best_model.pkl')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the JSON data from the request
#     data = request.json
#     if 'text' not in data:
#         return jsonify({'error': 'No text provided'}), 400

#     text = data['text']
    
#     # Clean and lemmatize the text
#     cleaned_text = clean_and_lemmatize_text([text])
    
#     # Make prediction
#     prediction = model.predict(cleaned_text)
    
#     # Return the result
#     return jsonify({'prediction': int(prediction[0])})

# # if __name__ == '__main__':
# #     app.run(debug=True)

from flask import Flask, request, jsonify
import joblib
import nltk
from text_cleaner import clean_and_lemmatize_text  # Ensure this module is in the same directory

# Initialize Flask app
app = Flask(__name__)

# Set NLTK data path to the location where you have the wordnet data
nltk_data_path = 'nltk_data'
nltk.data.path.append(nltk_data_path)

# Ensure the wordnet data is available
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_path)

# Load the model
try:
    model = joblib.load('best_model1.pkl')
except FileNotFoundError:
    raise RuntimeError("Model file 'best_model1.pkl' not found. Please ensure the file is in the correct directory.")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    
    # Clean and lemmatize the text
    cleaned_text = clean_and_lemmatize_text([text])
    
    # Make prediction
    try:
        prediction = model.predict(cleaned_text)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    # Return the result
    return jsonify({'prediction': int(prediction[0])})

# Run the app if this file is executed directly
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
