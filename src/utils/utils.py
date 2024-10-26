import torch 
from tqdm import tqdm
import json
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from models.model import BERTTicketClassifier
import wandb
import yaml

class SimpleLoss(torch.nn.Module):
    def __init__(self):
        super(SimpleLoss, self).__init__()

    def forward(self, probs, targets):
        return torch.nn.functional.cross_entropy(probs, targets)
    

def train_epoch(model, data, val_data, loss_fn, optimizer, scheduler, batch_size=8):
    model.train()
    batches = []
    for i in range(0, len(data), batch_size):
        if i + batch_size < len(data):
            batches.append(data[i:i+batch_size])
        else:
            batches.append(data[i:])
    total_loss = 0
    for batch in tqdm(batches, desc="Training", unit="batch"):
        optimizer.zero_grad()
        input_ids = torch.tensor(batch['input_ids']).to('cuda')
        input_ids = input_ids.view(len(batch['input_ids']), -1)
        attention_mask = torch.tensor(batch['attention_mask']).to('cuda')
        attention_mask = attention_mask.view(len(batch['input_ids']), -1)
        targets = torch.tensor(batch['sentiment']).to('cuda')
        probs = model(input_ids, attention_mask = attention_mask)
        loss = loss_fn(probs, targets)
        wandb.log({"Batch train loss": loss.item()})
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    eval_total_loss = 0
    with torch.no_grad():
        model.eval()
        eval_batches = []
        for i in range(0, len(val_data), batch_size):
            if i + batch_size < len(val_data):
                eval_batches.append(val_data[i:i+batch_size])
            else:
                eval_batches.append(val_data[i:])

        for batch in tqdm(eval_batches, desc="Validation", unit="batch"):
            input_ids = torch.tensor(batch['input_ids']).to('cuda')
            input_ids = input_ids.view(len(batch['input_ids']), -1)
            attention_mask = torch.tensor(batch['attention_mask']).to('cuda')
            attention_mask = attention_mask.view(len(batch['input_ids']), -1)
            targets = torch.tensor(batch['sentiment']).to('cuda')
            probs = model(input_ids, attention_mask = attention_mask)
            loss = loss_fn(probs, targets)
            wandb.log({"Batch eval loss": loss.item()})
            eval_total_loss += loss.item()

    model.train()
    return total_loss / len(batches), eval_total_loss / len(eval_batches)

def train_and_log(config = None):
    with wandb.init(project="bert-ticket-classifier", config=config):
        config = wandb.config
        category_mapping = load_category_mapping('data/processed/category_mapping.json')
        # load the BERT base model
        model, tokenizer = load_BERT_encoder(config.model_name)
        # load data for training
        raw_datasets, tokenized_datasets = prepare_training('data/processed/cleaned_tweets.csv', tokenizer)

        # initialize the model
        model = BERTTicketClassifier(config.model_name, len(category_mapping))
        model = model.to('cuda')
        loss_fn = SimpleLoss().to('cuda')
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        total_steps = len(tokenized_datasets['train']) // config.batch_size * config.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)
        wandb.watch(model, log="all")
        eval_losses = []
        for epoch in range(config.epochs):
            train_loss, eval_loss = train_epoch(model, tokenized_datasets['train'], tokenized_datasets['val'], loss_fn, optimizer, scheduler, batch_size=config.batch_size)
            eval_losses.append(eval_loss)
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "eval_loss": eval_loss
            })
    return model, train_loss, eval_losses





def load_category_mapping(mapping_path):
    with open(mapping_path, 'r') as f:
        category_mapping = json.load(f)
    return category_mapping

def load_BERT_encoder(model_name):
    model = BertModel.from_pretrained(model_name)
    model = model.to('cuda')
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer

def prepare_training(data_path, tokenizer, seed = 42):
    dataset = load_dataset('csv', data_files=data_path)
    # Split the dataset into train and test (90-10 split first)
    train_test_split = dataset['train'].train_test_split(test_size=0.2, shuffle=True, seed=seed)

    # Split the test set further into validation and test (50-50 split)
    val_test_split = train_test_split['test'].train_test_split(test_size=0.5, shuffle=True, seed=seed)

    # Combine the splits into a single DatasetDict
    dataset_split = {
        'train': train_test_split['train'],
        'val': val_test_split['train'],
        'test': val_test_split['test']
    }

    train_dataset = dataset_split['train']
    val_dataset = dataset_split['val']
    test_dataset = dataset_split['test']

    def tokenize_data(data):
        return tokenizer(data['text'], padding='max_length', truncation=True, max_length=256, return_tensors='pt')

    train_tok_dataset = train_dataset.map(tokenize_data)
    val_tok_dataset = val_dataset.map(tokenize_data)
    test_tok_dataset = test_dataset.map(tokenize_data)

    # remove the original text column
    train_tok_dataset = train_tok_dataset.remove_columns('text')
    val_tok_dataset = val_tok_dataset.remove_columns('text')
    test_tok_dataset = test_tok_dataset.remove_columns('text')

    # group the data into a DatasetDict
    tokenized_datasets = {
        'train': train_tok_dataset,
        'val': val_tok_dataset,
        'test': test_tok_dataset
    }

    return dataset_split, tokenized_datasets

def save_model(model, path):
    torch.save(model.state_dict(), path)

@torch.no_grad()
def evaluate(model, data, category_mapping, batch_size=32):
    with wandb.init(project='bert-ticket-classifier', job_type='testing'):
        model.eval()
        batches = []
        for i in range(0, len(data), batch_size):
            if i + batch_size < len(data):
                batches.append(data[i:i+batch_size])
            else:
                batches.append(data[i:])
        all_preds = []
        for batch in tqdm(batches, desc="Test", unit="batch"):
                input_ids = torch.tensor(batch['input_ids']).to('cuda')
                input_ids = input_ids.view(len(batch['input_ids']), -1)
                attention_mask = torch.tensor(batch['attention_mask']).to('cuda')
                attention_mask = attention_mask.view(len(batch['input_ids']), -1)
                targets = torch.tensor(batch['sentiment']).to('cuda')
                # print(len(targets))
                probs = model(input_ids, attention_mask = attention_mask)
                # extract prediction
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
        # calculate accuracy
        correct = 0
        for i in range(len(all_preds)):
            if all_preds[i] == data[i]['sentiment']:
                correct += 1
        wandb.log({
            "accuracy": correct / len(all_preds)
        })
        inverse_mapping = {v: k for k, v in category_mapping.items()}
        y_true = [data[i]['sentiment'] for i in range(len(data))]
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=y_true, preds=all_preds, class_names=["Positive", "Negative"])})
        wandb.log({"examples": wandb.Table(data=[[inverse_mapping[pred], inverse_mapping[truth]] for pred, truth in zip(all_preds, y_true)], columns=["Prediction", "Ground Truth"])})
        wandb.finish()
    return correct / len(all_preds), all_preds

def save_predictions(preds, targets, mapping, path):
    # create a csv file with 3 columns: text, prediction, target
    with open(path, 'w') as f:
        f.write('prediction,target\n')
        for i in range(len(preds)):
            f.write(f"{mapping[preds[i]]},{mapping[targets[i]['sentiment']]}\n")
    print(f"Predictions saved to {path}")
    return
        
