import torch
import torch.nn as nn
from transformers import BertModel


class FrozenBertClassifier(nn.Module):

    def __init__(self, num_labels):

        super().__init__()

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased",
            output_hidden_states=True
        )

        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_embedding)

        return logits, outputs.hidden_states