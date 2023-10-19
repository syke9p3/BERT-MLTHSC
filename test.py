import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define the model path
model_path = "checkpoint"  # Change this to the path where your model is saved

# Load the tokenizer
bert_model = "GKLMIP/bert-tagalog-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_model)

# Load the trained model
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def preprocess_text(text):
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors='pt')
    return encoding

text = "Ang kupal ng babae na to"
encoded_text = preprocess_text(text)

model.eval()
with torch.no_grad():
    outputs = model(**encoded_text)
    predictions = outputs.logits.sigmoid().tolist()[0]  # Apply sigmoid to get probabilities

threshold = 0.5
labels = ["Age", "Gender", "Physical", "Race", "Religion", "Others"]
predicted_labels = [(label, f"{pred*100:.2f}%") for label, pred in zip(labels, predictions) if pred >= threshold]

print("Input:", text)
print("Labels:")
for label, probability in predicted_labels:
    print(f"({label}, {probability})")