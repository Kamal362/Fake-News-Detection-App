# WordLens Fake News Detector

## Overview

WordLens is a web application designed to detect fake news articles. It uses a machine learning model to analyze the text of news articles and determine their authenticity. The app provides a confidence level for its predictions and suggests potential media sources that might report the news.

## Features

- **News Analysis**: Users can input a news article or snippet to analyze its authenticity.
- **Confidence Level**: The app displays a confidence level for each prediction.
- **Potential Sources**: Suggests possible media sources that might report the news.
- **User Interface**: A simple and intuitive interface with a loading animation during analysis.

## Installation
pip install required.txt

## creating and activation of virtual environment
  # creation
    python -m venv myvenv_wc_v2.0

1. bash
  
    # Activation
    python -m venv myvenv_wc_v2.0
    source myvenv_wc_v2.0/bin/activate

2. terminal/windows
   [nameofvenv]\Scripts\Activate

### Prerequisites

- Python 3.8 or higher
- Flask
- Pickle
- A trained machine learning model and vectorizer saved as `model.pkl` and `vectorizer.pkl`.

### Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd WordLens_V2.0

# run 
python app.py
