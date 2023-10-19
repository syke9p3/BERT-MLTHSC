import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel, AdamW, get_cosine_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss

class MLTHSClassifier(pl.LightningModule):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.threshold = config['threshold']
        self.bert = AutoModel.from_pretrained(config['model_name'], return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.config['n_labels'])
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.last_hidden_state[:, 0, :]
        cls_embedding = torch.sigmoid(cls_embedding)
        logits = self.classifier(cls_embedding)
        loss = 0
        if labels is not None:
            loss = self.criterion(
                        logits.view(-1, self.config['n_labels']), 
                        labels.view(-1, self.config['n_labels'])
                    )
        return loss, logits

    def calculate_metrics(self, labels, predictions, threshold):
        precision = precision_score(labels, (predictions > threshold).detach().cpu().numpy(), average=None)
        recall = recall_score(labels, (predictions > threshold).detach().cpu().numpy(), average=None)
        f1 = f1_score(labels, (predictions > threshold).detach().cpu().numpy(), average=None)
        h_loss = hamming_loss(labels, (predictions > threshold).detach().cpu().numpy())
        return precision, recall, f1, h_loss

    def log_metrics(self, metrics, prefix):
        for i, label_name in enumerate(self.labels):
            self.log(f'{prefix}_Precision_{label_name}', metrics[0][i], on_step=False, on_epoch=True)
            self.log(f'{prefix}_Recall_{label_name}', metrics[1][i], on_step=False, on_epoch=True)
            self.log(f'{prefix}_F1_{label_name}', metrics[2][i], on_step=False, on_epoch=True)
        self.log(f'{prefix}_Hamming_Loss', metrics[3], on_step=False, on_epoch=True)

    def training_step(self, batch, batch_index_):
        loss, logits = self(**batch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {
            'loss': loss,
            'predictions': logits,
            'labels': batch['labels']
        }

    def validation_step(self, batch, batch_index_):
        loss, logits = self(**batch)

        # Extract labels and predictions
        labels = batch['labels']
        threshold = self.config['threshold']

        # Calculate precision, recall, and F1 scores
        precision, recall, f1, hamming_loss = self.calculate_metrics(labels, logits, threshold)

        # Log precision, recall, and F1 scores per label
        self.log_metrics([precision, recall, f1, hamming_loss], prefix="Label")

        self.log('val_loss', loss, on_step=False, on_epoch=True)

        return {
            'val_loss': loss,
            'predictions': logits,
            'labels': labels
        }

    def test_step(self, batch, batch_index_):
        loss, logits = self(**batch)
        return logits

    def on_training_epoch_end(self, outputs, threshold):

        labels = []
        predictions = []

        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        precision, recall, f1, h_loss = self.calculate_metrics(labels, predictions, threshold)

        self.log('Precision_micro', precision, on_step=False, on_epoch=True)
        self.log('Recall_micro', recall, on_step=False, on_epoch=True)
        self.log('F1_micro', f1, on_step=False, on_epoch=True)
        self.log('Hamming_Loss_micro', h_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['w_decay'])
        total_steps = self.config['train_size'] / self.config['bs']
        warmup_steps = math.floor(total_steps * self.config['warmup'])
        scheduler = get_cosine_schedule_with_warmup(
                        optimizer, 
                        warmup_steps, 
                        total_steps
                    )
        return [optimizer], [scheduler]