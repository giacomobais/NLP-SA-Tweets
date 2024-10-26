import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from models.model import BERTTicketClassifier
from utils.utils import load_BERT_encoder, load_category_mapping
import yaml

app = FastAPI()

# Load configuration and model
config = yaml.safe_load(open('config/config.yaml'))
_, tokenizer = load_BERT_encoder(config['model_name']) 
category_mapping = load_category_mapping('data/processed/category_mapping.json')
inverse_category_mapping = {v: k for k, v in category_mapping.items()} 
model = BERTTicketClassifier(config['model_name'], len(category_mapping))
model.load_state_dict(torch.load("models/bert_ticket_classifier.pt"))
model.eval()

# Define request/response models
class Query(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    category: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
def predict(query: Query):
    # Tokenize input text and perform model prediction
    input_data = tokenizer(query.text, return_tensors='pt')
    with torch.no_grad():
        logits = model(input_data['input_ids'], attention_mask=input_data['attention_mask'])
    
    pred = torch.argmax(logits, dim=1).item()
    category = inverse_category_mapping[pred]
    confidence = logits[pred]
    
    return PredictionResponse(category=category, confidence=confidence)
