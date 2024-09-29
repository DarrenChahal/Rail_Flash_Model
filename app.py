from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import nltk
from text_cleaner import clean_and_lemmatize_text  # Ensure this module is in the same directory

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

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
    data = request.json
    
    # Change 'text' to 'grievanceDescription' to match the frontend
    if 'grievanceDescription' not in data:
        return jsonify({'error': 'No text provided'}), 400

    # Use the correct key to get the text
    text = data['grievanceDescription']
    
    # Clean and lemmatize the text
    cleaned_text = clean_and_lemmatize_text([text])

    try:
        prediction = model.predict(cleaned_text)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'prediction': int(prediction[0])})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
