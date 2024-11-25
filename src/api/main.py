import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pydantic import BaseModel
import torch
from src.models.model import BERTTicketClassifier
from src.utils.utils import load_BERT_encoder, load_category_mapping
import yaml
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

app = FastAPI()
# Mount static folder for CSS and other static files
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="src/api/templates")


# Load configuration and model
config = yaml.safe_load(open('config/config.yaml'))
_, tokenizer = load_BERT_encoder(config['model_name'], device='cpu') 
category_mapping = load_category_mapping('data/processed/category_mapping.json')
inverse_category_mapping = {v: k for k, v in category_mapping.items()} 
model = BERTTicketClassifier(config['model_name'], len(category_mapping))
model.load_state_dict(torch.load("models/bert_ticket_classifier.pt", map_location=torch.device('cpu')))
model.eval()

# Define request/response models
class Query(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    category: str
    confidence: float

# let's make a quick home page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/chat")
async def get_chat(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})

@app.post("/predict", response_model=PredictionResponse)
async def predict(query: Query):
    # Tokenize input text and perform model prediction
    input_data = tokenizer(query.text, return_tensors='pt')
    with torch.no_grad():
        logits = model(input_data['input_ids'], attention_mask=input_data['attention_mask'])
    
    pred = torch.argmax(logits, dim=1).item()
    # softmax the logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    # get the probabilty rounded to 4 decimal places
    confidence = round(probs[0][pred].item(), 4)*100
    category = category_mapping[str(pred)]
    
    return PredictionResponse(category=category, confidence=confidence)
