from fastapi import FastAPI
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained model
classifier = pipeline("text-classification", model="lvwerra/distilbert-imdb")

# API Route for Fake News Detection (Accepts URL Query Parameter)
@app.get("/predict/")
def detect_fake_news(text: str):
    result = classifier(text)
    label = result[0]['label']
    return {"text": text, "prediction": "Fake News" if label == 'NEGATIVE' else "Real News"}
  
