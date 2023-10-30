import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

bert_model = "gklmip/bert-tagalog-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_model)

import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('./dataset/mlthsc.csv', nrows=1000)

train_data, test_data = train_test_split(dataset, test_size=0.4, random_state=42) # 600, 400
test_data, val_data = train_test_split(test_data, test_size=0.25, random_state=42) # 300, 100

LABELS = ["Age", "Gender", "Physical", "Race", "Religion", "Others"]
id2label = {idx:label for idx, label in enumerate(LABELS)}
label2id = {label:idx for idx, label in enumerate(LABELS)}

def preprocess_data(data):
    text = data["Text"]

    encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128,
            return_token_type_ids=False,
            return_attention_mask=True
        )
    
    labels = data[LABELS]
    
    representation = {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'labels': torch.FloatTensor(labels)
    }

    return representation


from datasets import Dataset
import torch

# Create a list of encoded examples for train and test data
encoded_train_data = [preprocess_data(row) for _, row in train_data.iterrows()]
val_data = [preprocess_data(row) for _, row in val_data.iterrows()]

# Combine the encoded examples into a dictionary
encoded_train_dict = {key: [example[key] for example in encoded_train_data] for key in encoded_train_data[0]}
val_dict = {key: [example[key] for example in val_data] for key in val_data[0]}

# Convert the dictionaries to datasets
train_dataset = Dataset.from_dict(encoded_train_dict)
val_dataset = Dataset.from_dict(val_dict)

# Print the first few examples to verify the encoding
print(train_dataset)
print(val_dataset)

print(train_dataset[0])
print(val_dataset[0])

model = AutoModelForSequenceClassification.from_pretrained(bert_model,
                                                           problem_type="multi_label_classification",
                                                           num_labels=len(LABELS),
                                                           id2label=id2label,
                                                           label2id=label2id)

batch_size = 8

args = TrainingArguments(
    "checkpoint",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from transformers import EvalPrediction
import torch

def multilabel_metrics(predictions, labels, threshold=0.5):

    print("predictions:", predictions)

    # Apply sigmoid activation to logits/raw scores from the classifier 
    sigmoid = torch.nn.Sigmoid()
    probabilities = sigmoid(torch.Tensor(predictions))

    print("probabilities:", probabilities)

    # Filter out labels using the 0.5 threshold
    y_pred = np.zeros(probabilities.shape)
    y_pred[np.where(probabilities >= threshold)] = 1

    y_true = np.zeros(labels.shape)
    y_true[np.where(labels == 1)] = 1

    print("Y PRED:", y_pred)
    print("Y TRUE:", y_true)
    
    confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)
    print(confusion_matrix)
    label_metrics = {}
    
    classes = ['Age', 'Gender', 'Physical', 'Race', 'Religion', 'Others']

    for i in range(confusion_matrix.shape[0]):
        TP = confusion_matrix[i, 1, 1]  # True Positives
        FP = confusion_matrix[i, 0, 1]  # False Positives
        FN = confusion_matrix[i, 1, 0]  # False Negatives
        TN = confusion_matrix[i, 0, 0]  # True Negatives

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

        label_name = classes[i]

        label_metrics[label_name] = {
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1_score
        }

    # Calculate Hamming Loss
    xor_result = np.logical_xor(y_true, y_pred)
    xor_sum = np.sum(xor_result)
    hamming_loss = xor_sum / (y_true.shape[0] * y_true.shape[1])
    
    label_metrics['hamming_loss'] = 1 - hamming_loss

    print("Evaluation")
    print("Age")
    print("    Precision: ", label_metrics['Age']['Precision'])
    print("    Recall: ", label_metrics['Age']['Recall'])
    print("    F-Measure: ", label_metrics['Age']['F1-Score'])
    print("Gender")
    print("    Precision: ", label_metrics['Gender']['Precision'])
    print("    Recall: ", label_metrics['Gender']['Recall'])
    print("    F-Measure: ", label_metrics['Gender']['F1-Score'])
    print("Physical")
    print("    Precision: ", label_metrics['Physical']['Precision'])
    print("    Recall: ", label_metrics['Physical']['Recall'])
    print("    F-Measure: ", label_metrics['Physical']['F1-Score'])
    print("Race")
    print("    Precision: ", label_metrics['Race']['Precision'])
    print("    Recall: ", label_metrics['Race']['Recall'])
    print("    F-Measure: ", label_metrics['Race']['F1-Score'])
    print("Religion")
    print("    Precision: ", label_metrics['Religion']['Precision'])
    print("    Recall: ", label_metrics['Religion']['Recall'])
    print("    F-Measure: ", label_metrics['Religion']['F1-Score'])
    print("Others")
    print("    Precision: ", label_metrics['Others']['Precision'])
    print("    Recall: ", label_metrics['Others']['Recall'])
    print("    F-Measure: ", label_metrics['Others']['F1-Score'])
    print("\nHamming Loss: ", label_metrics['hamming_loss'])

    metrics = {
        'Age_precision':  label_metrics['Age']['Precision'],
        'Age_recall':  label_metrics['Age']['Recall'],
        'Age_f-measure':  label_metrics['Age']['F1-Score'],
        'Gender_precision':  label_metrics['Gender']['Precision'],
        'Gender_recall':  label_metrics['Gender']['Recall'],
        'Gender_f-measure':  label_metrics['Gender']['F1-Score'],
        'Physical_precision':  label_metrics['Physical']['Precision'],
        'Physical_recall':  label_metrics['Physical']['Recall'],
        'Physical_f-measure':  label_metrics['Physical']['F1-Score'],
        'Race_precision':  label_metrics['Race']['Precision'],
        'Race_recall':  label_metrics['Race']['Recall'],
        'Race_f-measure':  label_metrics['Race']['F1-Score'],
        'Religion_precision':  label_metrics['Religion']['Precision'],
        'Religion_recall':  label_metrics['Religion']['Recall'],
        'Religion_f-measure':  label_metrics['Religion']['F1-Score'],
        'Others_precision':  label_metrics['Others']['Precision'],
        'Others_recall':  label_metrics['Others']['Recall'],
        'Others_f-measure':  label_metrics['Others']['F1-Score'],
        'hamming_loss':  label_metrics['hamming_loss']
    }

    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

    print("preds", preds)

    result = multilabel_metrics(predictions=preds, labels=p.label_ids, threshold=0.5)
    return result



trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

import time

start_time = time.time()
print("Start time: ", start_time)
trainer.train()
end_time = time.time()
print("End time: ", end_time)


elapsed_time = end_time - start_time
print(f"Total training time elapsed: {elapsed_time} seconds")

trainer.save_model("model-trial-3")
print("Trainer Evaluate")

print(trainer.evaluate())
