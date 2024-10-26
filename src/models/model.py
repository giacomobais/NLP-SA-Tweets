import torch
from transformers import BertModel

class TicketEncoder(torch.nn.Module):
    def __init__(self, model):
        super(TicketEncoder, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask)
        cls = output.pooler_output
        return cls
    

class BERTTicketClassifier(torch.nn.Module):
    def __init__(self, model_name, num_classes):
        super(BERTTicketClassifier, self).__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.encoder = TicketEncoder(self.model)
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        cls = self.encoder(input_ids, attention_mask)
        logits = self.classifier(cls)
        return logits