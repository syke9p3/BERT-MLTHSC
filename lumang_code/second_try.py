import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import math

from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional.classification import hamming_distance, precision, recall, f1_score
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_cosine_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score

# import dataset files
train_path = './dataset/train.csv'
val_path = './dataset/val.csv'
test_path = './dataset/test.csv'
dataset_path = './dataset/mlthsc.csv'

data_frame = pd.read_csv(dataset_path)
# print(dataset_data)

# Inspect data # Todo: move to Dataset class

LABELS = ['Age', 'Gender', 'Physical', 'Race', 'Religion', 'Others']

print(data_frame[LABELS].sum())

# Plot dataset
plt.xlabel("Labels")
plt.ylabel("No. of instances")
plt.title("Labels")
data_frame[LABELS].sum().sort_values().plot(kind="barh")
# plt.show()

class MLTHS_Dataset(Dataset):

    def __init__(self, data_path, tokenizer, labels, max_token_len: int = 128, train_ratio=0.6, test_ratio=0.3):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_token_len = max_token_len
        # self.train_ratio = train_ratio
        # self.test_ratio = test_ratio
        self._prepare_data()

    def _prepare_data(self):
        data = pd.read_csv(self.data_path)
        self.data = data

        # --------------------- TODO: Normalize ------------------------------------------------

    def comment2(self):

        # print("prepare data")
        # print(data)

        # # Calculate the number of samples for training, testing, and validation
        # total_size = len(data)
        # train_size = int(total_size * self.train_ratio)
        # test_size = int(total_size * self.test_ratio)
        # val_size = total_size - train_size - test_size
        #
        # # Split the data into training, testing, and validation sets
        # train_data = data.sample(n=train_size, random_state=7)
        # remaining_data = data.drop(train_data.index)
        # test_data = remaining_data.sample(n=test_size, random_state=42)
        # val_data = remaining_data.drop(test_data.index)

        # self.train_data = train_data
        # self.test_data = test_data
        # self.val_data = val_data
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        text = str(item['Text'])
        labels = item[self.labels]

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(text,
                                            add_special_tokens=True,
                                            return_tensors='pt',
                                            padding='max_length',
                                            truncation=True,
                                            max_length=self.max_token_len,
                                            return_token_type_ids=False,
                                            return_attention_mask=True
                                            )

        # print(encoding["input_ids"].shape, encoding["attention_mask"].shape)

        # token_counts = []
        # for _, row in self.data.iterrows():
        #     token_count = len(tokenizer.encode(
        #         row["Text"],
        #         max_length=512,
        #         truncation=True
        #     ))
        #     token_counts.append(token_count)
        # sns.histplot(token_counts)
        # plt.xlim([0, 128])

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }

bert_model = "gklmip/bert-tagalog-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_model)

mlths_ds_train = MLTHS_Dataset(train_path, tokenizer, labels=LABELS)
mlths_ds_val = MLTHS_Dataset(val_path, tokenizer, labels=LABELS)

def comment():
    return 0
    # mlths_ds_test = MLTHS_Dataset(test_path, tokenizer, labels=LABELS)

    # print(dataset_data.iloc[0])
    # print(mlths_ds_train.__getitem__(0))

    # train_ids = set(mlths_ds_train.data['ID'])
    # val_ids = set(mlths_ds_val.data['ID'])

    # overlapping_ids = train_ids.intersection(val_ids)
    #
    # if not overlapping_ids:
    #     print("No overlapping IDs between training and validation datasets.")
    # else:
    #     print("There are overlapping IDs between training and validation datasets.")

# Data Module

class MLTHS_Data_Module(pl.LightningDataModule):

    def __init__(self, train_path, val_path, test_path, labels, model_name, batch_size: int = 8, max_token_len: int = 128):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.labels = labels
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = MLTHS_Dataset(self.train_path, self.tokenizer, self.labels)
            self.val_dataset = MLTHS_Dataset(self.val_path, self.tokenizer, self.labels)
            self.test_dataset = MLTHS_Dataset(self.test_path, self.tokenizer, self.labels)

        if stage == 'predict':
            self.val_dataset = MLTHS_Dataset(self.val_path, self.tokenizer, self.labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2, shuffle=False, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2, shuffle=False)
        
        
# Model

class MLTHSClassifier(pl.LightningModule):

    def __init__(self, config: dict, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.config = config
        self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict=True)
        self.hidden = nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
        self.classifier = nn.Linear(self.pretrained_model.config.hidden_size, self.config['n_labels'])
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.classifier.weight)
        self.loss_function = nn.BCEWithLogitsLoss(reduction='mean')
        self.dropout = nn.Dropout()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.last_hidden_state[:, 0, :]
        cls_embedding = F.sigmoid(cls_embedding)
        logits = self.classifier(cls_embedding)
        loss = 0
        if labels is not None:
            loss = self.loss_function(logits.view(-1, self.config['n_labels']), labels.view(-1, self.config['n_labels']))
        return loss, logits

    def training_step(self, batch, batch_index_):
        loss, logits = self(**batch)
        self.log("train loss", loss, prog_bar=True, logger=True)
        return {
            'loss': loss,
            'predictions': logits,
            'labels': batch['labels']
        }

    def validation_step(self, batch, batch_index_):
        loss, logits = self(**batch)
        self.log("validation loss", loss, prog_bar=True, logger=True)
        return {
            'val_loss': loss,
            'predictions': logits,
            'labels': batch['labels']
        }

    def test_step(self, batch, batch_index_):
        loss, logits = self(**batch)
        return logits

    def on_training_epoch_end(self, logits):

        labels = []
        predictions = []

        for logit in logits:
            for out_labels in logit["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in logit["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)


    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['w_decay'])
        total_steps = self.config['train_size'] / self.config['bs']
        warmup_steps = math.floor(total_steps * self.config['warmup'])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [scheduler]


# Trainer

mlths_data_module = MLTHS_Data_Module(train_path, val_path, test_path, labels=LABELS, model_name=bert_model)
mlths_data_module.setup()
dl = mlths_data_module.train_dataloader()

print("len(dl)")
print(len(dl))


if __name__ == '__main__':
    config = {
        'model_name': bert_model,
        'n_labels': len(LABELS),
        'bs': 8,
        'lr': 2e-5,
        'warmup': 0.2,
        'train_size': len(dl),
        'w_decay': 0.01,
        'n_epochs': 5
    }

    mlths_data_module = MLTHS_Data_Module(train_path, val_path, test_path, labels=LABELS, model_name=bert_model, batch_size=config['bs'])
    mlths_data_module.setup()
    dl = mlths_data_module.train_dataloader()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = MLTHSClassifier(config)
    idx = 0
    input_ids = mlths_ds_train.__getitem__(idx)['input_ids']
    attention_mask = mlths_ds_train.__getitem__(idx)['attention_mask']
    LABELS = mlths_ds_train.__getitem__(idx)['labels']
    loss, output = model(input_ids.unsqueeze(dim=0), attention_mask.unsqueeze(dim=0), LABELS.unsqueeze(dim=0))
    print(loss, output)
    print(LABELS.shape, output.shape, output)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"{config['n_epochs']}-val_loss:.2f",
        save_top_k=1,
        verbose=True,
        monitor="validation loss",
        mode="min"
    )

    logger = TensorBoardLogger("lightning_logs", name="hate-speech")

    trainer = pl.Trainer(max_epochs=config['n_epochs'], num_sanity_val_steps=2, log_every_n_steps=5)
    trainer.fit(model, mlths_data_module)

    trainer.test()

    trained_model = MLTHSClassifier.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        n_labels=len(LABELS)
    )

    trained_model.eval()
    trained_model.freeze()

    test_comment = "Masyadong babae tong mga iglesia na ito"
    encoding = tokenizer.encode_plus(
      test_comment,
      add_special_tokens=True,
      max_length=128,
      return_token_type_ids=False,
      padding="max_length",
      return_attention_mask=True,
      return_tensors='pt',
    )
    _, test_prediction = trained_model(encoding["input_ids"], encoding["attention_mask"])
    test_prediction = test_prediction.flatten().numpy()
    for label, prediction in zip(LABELS, test_prediction):
      print(f"{label}: {prediction}")