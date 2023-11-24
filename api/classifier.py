import os
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import re
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('gklmip/bert-tagalog-base-uncased')
model_name = "gklmip/bert-tagalog-base-uncased"
LABELS = ['Age', 'Gender', 'Physical', 'Race', 'Religion', 'Others']

class HateSpeechClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(HateSpeechClassifier, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert_model.config.hidden_size, num_labels)

        for param in self.bert_model.parameters():
            param.requires_grad = True

    def forward(self, ids, mask):
        bert_outputs = self.bert_model(ids, attention_mask=mask)
        cls_hidden_state = bert_outputs.last_hidden_state[:, 0, :] 
        dropped_out = self.dropout(cls_hidden_state)
        logits = self.linear(dropped_out)
        return logits
    
trained_model = HateSpeechClassifier(model_name, len(LABELS))
trained_model.load_state_dict(torch.load('D:\Repo\Thesis\BERT-MLTHSC\\api\good_model.pth'))

def preprocess_text(text):
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors='pt')
    return encoding

def get_predictions(test_sentence):

    encoded_test_sentence = preprocess_text(test_sentence)

    with torch.no_grad():
        logits = trained_model(ids=encoded_test_sentence['input_ids'], mask=encoded_test_sentence['attention_mask'])

    predictions = logits.flatten().sigmoid()  # Apply sigmoid to get probabilities

    # Get all labels
    label_probabilities = [{"name": label, "probability": f"{prob * 100:.2f}%"} for label, prob in zip(LABELS, predictions)]

    # Sort label probabilities in descending order
    label_probabilities = sorted(label_probabilities, key=lambda item: -float(item["probability"][:-1]))
    print(label_probabilities)

    threshold = 0.5

    # Labels greater than 0.5 threshold
    predicted_labels = [(label, f"{pred*100:.2f}%") for label, pred in zip(LABELS, predictions) if pred >= threshold]
    print("Input:", test_sentence)
    print("Probabilities: ", label_probabilities)

    print("Labels:")
    for label, probability in predicted_labels:
        print(f"({label}, {probability})")


    return label_probabilities

test_sentence = "Sana naman maintindihan mo kung gaano kahirap makipag-away "
