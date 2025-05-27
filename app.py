import os
from flask import Flask, render_template, request
import pickle
from pathlib import Path

app = Flask(__name__)

# Configure absolute paths
BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / "templates"

def load_ml_components():
    try:
        with open(BASE_DIR / "model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(BASE_DIR / "vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        print("ML components loaded successfully")
        return model, vectorizer
    except Exception as e:
        print(f"Error loading ML components: {e}")
        return None, None

model, vectorizer = load_ml_components()

@app.route('/', methods=['GET'])
def home():
    # Verify template exists
    if not (TEMPLATE_DIR / "index.html").exists():
        return "Template missing: Please ensure index.html exists in templates folder", 500
    
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Template rendering error: {e}")
        return f"Error rendering template: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    if not all([model, vectorizer]):
        return "ML service unavailable", 503
        
    try:
        news = request.form.get('news', '')
        if not news.strip():
            return "No text provided", 400
            
        vect_text = vectorizer.transform([news])
        prediction = model.predict(vect_text)[0]
        confidence = max(model.predict_proba(vect_text)[0])
        
        return render_template('index.html',
                            prediction="Fake" if prediction else "Real",
                            confidence=f"{confidence:.1%}",
                            sources=["Source A", "Source B", "Source C"])
    except Exception as e:
        print(f"Prediction error: {e}")
        return f"Prediction failed: {str(e)}", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)