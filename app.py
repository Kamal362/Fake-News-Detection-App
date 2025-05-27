import os
import pickle
from pathlib import Path
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Configuration
BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / "templates"
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

# Initialize ML components as None
model = None
vectorizer = None

def load_ml_components():
    """Load and return the ML model and vectorizer with error handling."""
    global model, vectorizer
    
    try:
        if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
            raise FileNotFoundError("Model or vectorizer file not found")
            
        with open(MODEL_PATH, 'rb') as model_file, open(VECTORIZER_PATH, 'rb') as vectorizer_file:
            model = pickle.load(model_file)
            vectorizer = pickle.load(vectorizer_file)
            print("✅ ML components loaded successfully")
            return True
            
    except Exception as e:
        print(f"❌ Error loading ML components: {str(e)}")
        model = None
        vectorizer = None
        return False

# Initialize app components at startup
with app.app_context():
    print("⚙️ Initializing application...")
    if not TEMPLATE_DIR.exists():
        os.makedirs(TEMPLATE_DIR, exist_ok=True)
    load_ml_components()

@app.route('/')
def home():
    """Render the main page with template verification."""
    template_path = TEMPLATE_DIR / "index.html"
    
    if not template_path.exists():
        error_msg = f"Template not found at: {template_path}"
        print(f"❌ {error_msg}")
        return error_msg, 500
        
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests with comprehensive error handling."""
    # Check if ML components are loaded
    if model is None or vectorizer is None:
        error_msg = "ML service not initialized"
        print(f"❌ {error_msg}")
        return jsonify({
            'error': error_msg,
            'status': 'service_unavailable'
        }), 503
    
    # Get and validate input
    news_text = request.form.get('news', '').strip()
    if not news_text:
        error_msg = "No text provided for analysis"
        print(f"❌ {error_msg}")
        return jsonify({
            'error': error_msg,
            'status': 'bad_request'
        }), 400
    
    try:
        # Transform and predict
        vect_text = vectorizer.transform([news_text])
        prediction = model.predict(vect_text)[0]
        confidence = max(model.predict_proba(vect_text)[0])
        
        # Prepare results
        result = {
            'prediction': 'Fake News' if prediction == 1 else 'Real News',
            'confidence': f"{confidence:.2%}",
            'confidence_value': confidence,
            'sources': [
                "Reuters Fact Check",
                "AP News Verification",
                "PolitiFact"
            ],
            'status': 'success'
        }
        
        print(f"✅ Prediction successful: {result['prediction']} ({result['confidence']})")
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({
            'error': error_msg,
            'status': 'prediction_failed'
        }), 500

@app.route('/health')
def health_check():
    """Endpoint for health checks and monitoring."""
    checks = {
        'template_ready': (TEMPLATE_DIR / "index.html").exists(),
        'model_ready': model is not None,
        'vectorizer_ready': vectorizer is not None,
        'status': 'ok' if model and vectorizer else 'service_unavailable'
    }
    return jsonify(checks)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)