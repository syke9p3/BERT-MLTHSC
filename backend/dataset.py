import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class MLTHSDataset(Dataset):

    def __init__(self, data: pd.DataFrame, tokenizer, labels: list, max_token_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_token_len = max_token_len
        self._preprocess_data()

    def _preprocess_data(self):
        return 0
        # TODO: add normalizer / preprocessor logic here  -----------------------------------------------------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        hate_speech_text = data_row.Text
        labels = data_row[self.labels]

        encoding = self.tokenizer.encode_plus(
            hate_speech_text,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            return_attention_mask=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(labels)
        }
    
    def _get_stats(self, _print: str = False):
        stats = {
            "text_count": len(self.data),
            "instance_per_label": self.data[self.labels].sum(),
            "shape": self.data.shape
        }
        if (_print): 
            print("\nDATASET STATISTICS:\n")
            print("Number of Text", len(self.data))
            print("Instance per Label\n", self.data[self.labels].sum())
            print("Shape: ", self.data.shape)
        return stats

    
    def _print_sample_hate_speech(self, index: int = 0, get_encoding: bool = False):
        sample_row = self.data.iloc[index]
        sample_text = sample_row.Text
        sample_labels = sample_row[self.labels]
        print("\nSAMPLE TRAINING HATE SPEECH:")
        print("Index: ", index)
        print("Text: ", sample_text)
        print("Labels: ", sample_labels.to_dict())

        encoding = self.tokenizer.encode_plus(
            sample_text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt',
        )
    
        if (get_encoding):
            print("Encoding:\n", encoding, "\n")
            print("Input IDs: ", encoding["input_ids"].squeeze()[:20])
            print("Attention Mask: ", encoding["attention_mask"].squeeze()[:20])
            print("Tokens:",  self.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze())[:20])

    def _get_data_frame(self, _print: str = False):
        if (_print): print(self.data)
        return self.data
    
class MLTHSDataModule(pl.LightningDataModule):
    
    def __init__(self, train_df, val_df, test_df, labels, tokenizer, batch_size=8, max_token_len=128):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.labels = labels
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.mlths_train_dataset = MLTHSDataset(self.train_df, self.labels)
        self.mlths_val_dataset = MLTHSDataset(self.val_df, self.labels)
        self.mlths_test_dataset = MLTHSDataset(self.test_df, self.labels)

    def train_dataloader(self):
        return DataLoader(self.mlths_train_dataset, batch_size=self.batch_size, num_workers=2, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mlths_val_dataset, batch_size=self.batch_size, num_workers=2, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.mlths_test_dataset, batch_size=self.batch_size, num_workers=2, shuffle=False)
    
