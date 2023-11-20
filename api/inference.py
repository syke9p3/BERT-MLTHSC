import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_checkpoint = "model-trial-3"
trained_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained("gklmip/bert-tagalog-base-uncased")

# test_sentence = "ambobo ng mga batang katoliko na bisaya"
LABELS = ["Age", "Gender", "Physical", "Race", "Religion", "Others"]

def preprocess_text(text):
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors='pt')
    return encoding

def get_predictions(test_sentence):

    encoded_test_sentence = preprocess_text(test_sentence)

    with torch.no_grad():
        model_outputs = trained_model(**encoded_test_sentence)

    predictions = model_outputs.logits.sigmoid().tolist()[0]  # Apply sigmoid to get probabilities

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