import requests

# URL of the Flask endpoint
url = 'http://127.0.0.1:5000/predict'  # Ensure this matches your Flask server URL

# Data to send for prediction
data = {'text': 'I am very dissatisfied with the cleanliness of the train.'}

# Make POST request
try:
    response = requests.post(url, json=data)
    response.raise_for_status()  # Raise an exception for HTTP errors
    # Print response
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
