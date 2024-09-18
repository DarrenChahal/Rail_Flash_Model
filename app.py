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
from text_cleaner import clean_and_lemmatize_text  # Import from your module

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = joblib.load('best_model1.pkl')

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
    prediction = model.predict(cleaned_text)
    
    # Return the result
    return jsonify({'prediction': int(prediction[0])})

# if __name__ == '__main__':
#     app.run(debug=True)
