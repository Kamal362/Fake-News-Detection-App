from flask import Flask, render_template, request
import pickle
import os
from typing import Tuple, Dict, Any

app = Flask(__name__)

# Constants
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

def load_ml_components() -> Tuple[Any, Any]:
    """Load machine learning model and vectorizer."""
    try:
        with open(MODEL_PATH, "rb") as model_file, open(VECTORIZER_PATH, "rb") as vectorizer_file:
            model = pickle.load(model_file)
            vectorizer = pickle.load(vectorizer_file)
            print("Model and vectorizer loaded successfully.")
            return model, vectorizer
    except Exception as e:
        print(f"Error loading ML components: {e}")
        raise

# Load components at startup
try:
    model, vectorizer = load_ml_components()
except Exception as e:
    print(f"Failed to initialize ML components: {e}")
    model, vectorizer = None, None

@app.route('/')
def home():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    if model is None or vectorizer is None:
        return render_template('index.html', 
                            prediction_text="Service unavailable", 
                            error="ML model not loaded")

    try:
        news_text = request.form.get('news', '')
        if not news_text.strip():
            return render_template('index.html', 
                                prediction_text="Invalid input", 
                                error="Empty news text provided")

        # Make prediction
        vect_text = vectorizer.transform([news_text])
        prediction = model.predict(vect_text)[0]
        confidence = max(model.predict_proba(vect_text)[0])
        
        # Prepare response
        sources = ["Source A", "Source B", "Source C"]  # Replace with actual sources
        result = {
            "prediction": "Fake News" if prediction == 1 else "Real News",
            "confidence": f"{confidence:.2%}",
            "sources": sources,
            "input_text": news_text  # Echo back the input
        }
        
        return render_template('index.html', **result)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template('index.html', 
                            prediction_text="Prediction error",
                            error=str(e))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))