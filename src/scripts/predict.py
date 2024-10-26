import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import BERTTicketClassifier
from utils.utils import load_BERT_encoder, load_category_mapping
import torch
import yaml


if __name__ == '__main__':
    # query the user for a sentence
    sentence = input("Enter a sentence: ")
    config = yaml.safe_load(open('config/config.yaml'))
    _, tokenizer = load_BERT_encoder(config['model_name']) 
    category_mapping = load_category_mapping('data/processed/category_mapping.json')
    inverse_category_mapping = {v: k for k, v in category_mapping.items()} 
    model = BERTTicketClassifier(config['model_name'], len(category_mapping))
    model.load_state_dict(torch.load("models/bert_ticket_classifier.pt"))
    model.eval()
    input_data = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        logits = model(input_data['input_ids'], attention_mask=input_data['attention_mask'])
    
    pred = torch.argmax(logits, dim=1).item()
    # softmax the logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    confidence = probs[0][pred].item()
    print(category_mapping[str(pred)], confidence)
