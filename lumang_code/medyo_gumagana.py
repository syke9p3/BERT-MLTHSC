"""

===================== FINE TUNING BERT: General Steps =================================================
    
    1. load dataset
    2. load transformer
    3. train
    4. evaluate

"""

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
from sklearn.metrics import multilabel_confusion_matrix


bert_model = "gklmip/bert-tagalog-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_model)
dataset_name = "syke9p3/multilabel-tagalog-hate-speech"
f_dataset_name = f"syke9p3/multilabel-tagalog-hate-speech"
dataset = load_dataset(dataset_name)

# Extract labels
labels = [label for label in dataset['train'].features.keys() if label not in ['Text'] and label not in ['ID']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}


print("id2label:", id2label)
print("label2id:", label2id)

# example tweet from dataset
print(dataset)
print("\nLabels: of " + dataset_name)
print(label2id)

# example tweet from dataset
tweet = dataset['train'][0]

print("Tweet: ", tweet)


def preprocess_data(examples):
  text = examples["Text"]
  encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
  labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
  labels_matrix = np.zeros((len(text), len(labels)))
  for idx, label in enumerate(labels):
    labels_matrix[:, idx] = labels_batch[label]

  encoding["labels"] = labels_matrix.tolist()

  return encoding

encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
example = encoded_dataset['train'][0]
print(example.keys())
print(tokenizer.decode(example['input_ids']))
print(example['labels'])
print([id2label[idx] for idx, label in enumerate(example['labels']) if label == 1.0])

encoded_dataset.set_format("torch")


model = AutoModelForSequenceClassification.from_pretrained(bert_model,
                                                           problem_type="multi_label_classification",
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)


batch_size = 8
metric_name = "hamming_loss"

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
    metric_for_best_model=metric_name,
)

from sklearn.metrics import hamming_loss, precision_score, recall_score, f1_score

# def compute_metrics(p: EvalPrediction):
#     preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
#     y_true = p.label_ids
#     sigmoid = torch.nn.Sigmoid()
#     probs = sigmoid(torch.Tensor(preds))
#     y_pred = np.zeros(probs.shape)
#     y_pred[np.where(probs >= 0.5)] = 1
#
#     metrics_per_label = {}
#     for label_idx, label_name in enumerate(labels):
#         y_true_label = y_true[:, label_idx]
#         y_pred_label = y_pred[:, label_idx]
#         precision = precision_score(y_true_label, y_pred_label, zero_division=1)
#         recall = recall_score(y_true_label, y_pred_label, zero_division=1)
#         f1 = f1_score(y_true_label, y_pred_label, zero_division=1)
#         metrics_per_label[label_name] = {
#             "precision": precision,
#             "recall": recall,
#             "f1": f1
#         }
#
#     hamming_loss_value = hamming_loss(y_true, y_pred)
#
#     return {
#         "f1": f1_score(y_true, y_pred, average='micro', zero_division=1),
#         "accuracy": accuracy_score(y_true, y_pred),
#         "hamming_loss": hamming_loss_value,
#         "metrics_per_label": metrics_per_label
#     }

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, hamming_loss
from transformers import EvalPrediction
import torch

def multi_labels_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    #f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')

    print("y pred shape: " ,y_pred.shape)
    print("Y PREDS: \n" ,y_pred)

    print("y true shape: " ,y_true.shape)
    print("Y TRUE: \n" ,y_true)

    
    confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)  
    print(confusion_matrix)

    accuracy = accuracy_score(y_true, y_pred)
    hamming = hamming_loss(y_true, y_pred)


    # return as dictionary
    metrics = {
               'hamming_loss': hamming,
               'accuracy': accuracy
    }

    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    result = multi_labels_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
print(trainer.evaluate())
result = trainer.evaluate()


trainer.save_model("MLTHSC_BERT")
print("Trainer Evaluate")


formatted_result = (
    f"Evaluation Metrics:\n"
    f"  Loss: {result['eval_loss']:.4f}\n"
    f"  F1: {result['eval_f1']:.2%}\n"
    f"  Accuracy: {result['eval_accuracy']:.2%}\n"
    f"  Hamming Loss: {result['eval_hamming_loss']:.2%}\n"
    "\nMetrics per label:\n"
)

for label, metrics in result['eval_metrics_per_label'].items():
    formatted_result += f"{label}:\n"
    formatted_result += f"  Precision: {metrics['precision']:.2%}\n"
    formatted_result += f"  Recall: {metrics['recall']:.2%}\n"
    formatted_result += f"  F1: {metrics['f1']:.2%}\n"

formatted_result += (
    f"\nTraining Stats:\n"
    f"  Runtime: {result['eval_runtime']:.2f} seconds\n"
    f"  Samples per second: {result['eval_samples_per_second']:.2f}\n"
    f"  Steps per second: {result['eval_steps_per_second']:.2f}\n"
    f"  Epoch: {result['epoch']}"
)

print(formatted_result)

from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


predictions = trainer.predict(encoded_dataset["test"]).predictions
y_true = encoded_dataset["test"]["labels"]

# Get the number of labels
num_labels = len(labels)

predictions_binary_list = []


for label_idx in range(num_labels):
    label_predictions = np.round(predictions[:, label_idx])
    label_predictions_binary = label_binarize(label_predictions, classes=[0, 1])
    predictions_binary_list.append(label_predictions_binary)

# Calculate the multi-label confusion matrices for each label
multilabel_conf_matrices = []
for label_idx in range(num_labels):
    conf_matrix = multilabel_confusion_matrix(y_true[:, label_idx], predictions_binary_list[label_idx])
    multilabel_conf_matrices.append(conf_matrix)


# Define a custom function to plot the confusion matrix
def plot_confusion_matrices(conf_matrices, label_names):
    num_labels = len(label_names)
    plt.figure(figsize=(10, 8 * num_labels))

    for i, (matrix, label) in enumerate(zip(conf_matrices, label_names)):
        plt.subplot(num_labels, 1, i + 1)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix for Label \'{label}\'')

    plt.tight_layout()
    plt.show()


# Get the label names
label_names = list(label2id.keys())

# Plot the multi-label confusion matrices
plot_confusion_matrices(multilabel_conf_matrices, label_names)