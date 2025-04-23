from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model
classifier = pipeline("text-classification", model="lvwerra/distilbert-imdb")

# API Route for Fake News Detection
@app.get("/predict/")
def detect_fake_news(text: str):
    result = classifier(text)
    label = result[0]['label']
    return {"text": text, "prediction": "Fake News" if label == 'NEGATIVE' else "Real News"}


