import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import BERTTicketClassifier
from utils.utils import train_and_log, save_model
import wandb
import yaml



if __name__ == '__main__':
    # get the main arguments
    sweep = eval(sys.argv[1])
    # load the config
    config = yaml.safe_load(open('config/config.yaml'))
    if sweep:
        sweep_config = yaml.safe_load(open('config/sweep_config.yaml'))
        sweep_id = wandb.sweep(sweep_config, project="bert-ticket-classifier")
        wandb.agent(sweep_id, train_and_log, count=5)
        wandb.finish()
    else:
        print('Standard training...')
        model, train_losses, eval_losses = train_and_log(config=config)
        wandb.finish()
        # save the model
        save_model(model, 'models/bert_ticket_classifier.pth')