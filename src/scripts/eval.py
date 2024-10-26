import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import BERTTicketClassifier
from transformers import BertTokenizer
from utils.utils import evaluate, load_category_mapping, prepare_training, save_predictions
import torch
import yaml

if __name__ == '__main__':
    # load the config
    config = yaml.safe_load(open('config/config.yaml'))
    # load the category mapping
    category_mapping = load_category_mapping('data/processed/category_mapping.json')
    inverted_mapping = {v: k for k, v in category_mapping.items()}
    # load the saved model
    tokenizer = BertTokenizer.from_pretrained(config['model_name'])
    model = BERTTicketClassifier(config['model_name'], len(category_mapping))
    model = model.to('cuda')
    model.load_state_dict(torch.load('models/bert_ticket_classifier.pt'))

    # load data for evaluation
    raw_datasets, tokenized_datasets = prepare_training('data/processed/cleaned_tweets.csv', tokenizer)
    print(tokenized_datasets['test'])
    accuracy, preds = evaluate(model, tokenized_datasets['test'], batch_size=config['batch_size'])
    print(f"Accuracy: {accuracy}")

    # save the predictions
    save_predictions(preds, tokenized_datasets['test'], inverted_mapping, 'data/outputs/predictions.csv')