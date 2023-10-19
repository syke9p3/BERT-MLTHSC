import torch
from transformers import AutoTokenizer, AutoModel
import gradio as gr

# Define the model path
model_path = "MLTHSC_BERT"  # Change this to the path where your model is saved

# Load the tokenizer
bert_model = "gklmip/bert-tagalog-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_model)

# Load the trained model
model = AutoModel.from_pretrained(model_path)

def preprocess_text(text):
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors='pt')
    return encoding

def predict_labels(text):
    encoded_text = preprocess_text(text)

    model.eval()
    with torch.no_grad():
        outputs = model(**encoded_text)
        probs = [str(round(pred * 100, 2)) + "%" for pred in outputs.logits.sigmoid().tolist()[0]]

    threshold = 0.5
    labels = ["Age", "Gender", "Physical", "Race", "Religion", "Others"]
    predicted_labels = list(zip(labels, probs))

    # Sort labels by the highest probability in descending order
    predicted_labels.sort(key=lambda x: float(x[1][:-1]), reverse=True)

    return predicted_labels

print(predict_labels("Ang mga batang babae sa kanto ay kung anu-ano ginagawa"))

iface = gr.Interface(
    fn=predict_labels,
    inputs=gr.Textbox(label="Input Text"),
    outputs=[gr.HighlightedText(label="Highlight")],
    live=True,
    title="Tagalog Hate Speech Classifier",
    description="Enter a hate speech in Tagalog to classify its targets: Age, Gender, Physical, Race, Religion, Others.",
)
iface.launch()