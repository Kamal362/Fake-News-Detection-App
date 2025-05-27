from flask import Flask, render_template, request
import pickle

app = Flask(__name__, template_folder='templates')

# Load model and vectorizer
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    print("Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None, confidence=0, sources=[], news="")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        news = request.form['news']
        vect_text = vectorizer.transform([news])
        prediction = model.predict(vect_text)[0]
        confidence = max(model.predict_proba(vect_text)[0])  # Get the confidence level
        sources = ["Source A", "Source B", "Source C"]  # Example sources
        #print(f'Received news: {news}')
        label = "Fake News" if prediction == 1 else "Real News"
        return render_template('index.html', prediction_text=label, confidence=confidence, sources=sources, news = news)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', prediction_text="Error in prediction")
    # try:
    #     news = request.form['news']
    #     print(f"Received news: {news}")

    #     vect_text = vectorizer.transform([news])
    #     print(f"Vectorized text: {vect_text}")

    #     prediction = model.predict(vect_text)[0]
    #     print(f"Prediction: {prediction}")

    #     label = "Fake News" if prediction == 1 else "Real News"
    #     return render_template('index.html', prediction_text=label)
    # except Exception as e:
    #     print(f"Error during prediction: {e}")
    #     return render_template('index.html', prediction_text="Error in prediction")

if __name__ == "__main__":
    app.run(debug=True)