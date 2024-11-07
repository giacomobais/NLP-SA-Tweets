import sys


from src.models.model import BERTTicketClassifier
# do the same import but make it absolute


if __name__ == '__main__':
    model = BERTTicketClassifier()
    model.train()