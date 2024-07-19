import torch
from torch import nn


class CustomFinBert(nn.Module):
    def __init__(self, original_bert, n_numeric_feats, n_classes, hidden_dims=[]):
        super().__init__()
        classifier_nfeats =  original_bert.classifier.in_features + n_numeric_feats
        self.bert = original_bert.bert
        self.dropout = original_bert.dropout
        classifier_layers = []
        prev_dim = classifier_nfeats
        for dim in hidden_dims:
            classifier_layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        classifier_layers.append(nn.Linear(prev_dim, n_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, input_ids, attention_mask, token_type_ids, numeric_feats):
        bert_out = self.dropout(self.bert(input_ids, attention_mask, token_type_ids)[1])
        classifier_input = torch.cat([bert_out, numeric_feats], dim=-1)
        logits = self.classifier(classifier_input)
        return logits
