import pandas as pd
from transformers import AutoTokenizer, AutoModel
from dataset import MLTHSDataModule
from classifier import MLTHSClassifier

model_name = "gklmip/bert-tagalog-base-uncased"
BERT_MODEL = AutoModel.from_pretrained(model_name, return_dict=True)
BERT_TOKENIZER = AutoTokenizer.from_pretrained(model_name)

train_path = './dataset/train.csv'
val_path = './dataset/val.csv'
test_path = './dataset/test.csv'
dataset_path = './dataset/mlthsc.csv'

df = pd.read_csv(dataset_path)
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)
LABELS = ['Age', 'Gender', 'Physical', 'Race', 'Religion', 'Others']

# plt.xlabel("Labels")
# plt.ylabel("No. of instances")
# plt.title("Labels")
# df[LABELS].sum().sort_values().plot(kind="barh")
# plt.show()

# token_counts = []
# 
# for _, row in df.iterrows():
#     token_count = len(BERT_TOKENIZER.encode(
#         row["Text"], 
#         max_length=128, 
#         truncation=True
#     )
# )
#     token_counts.append(token_count)

# sns.histplot(token_counts)
# plt.xlim([0, 128])
# plt.show()




if __name__ == '__main__':
    
    # mlths_train_dataset._get_stats(_print=True)
    # mlths_train_dataset._print_sample_hate_speech(index = 17, get_encoding=True)

    # print(mlths_train_dataset[17]["labels"])
    # print(mlths_train_dataset[17]["input_ids"].shape)

    # sample_batch = next(iter(DataLoader(mlths_train_dataset, batch_size=8, num_workers=2)))
    # print(sample_batch["input_ids"].shape, sample_batch["attention_mask"].shape)

    N_EPOCHS = 5
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    THRESHOLD = 0.5

    mlths_data_module = MLTHSDataModule(
        train_df,
        val_df,
        test_df, 
        tokenizer=BERT_TOKENIZER,
        batch_size=BATCH_SIZE,
        max_token_len=128
    )

    mlths_data_module.setup()
    mlths_dl = mlths_data_module.train_dataloader()

    config = {
        'model_name': model_name,
        'n_labels': len(LABELS),
        'train_size': len(mlths_dl),
        'bs': BATCH_SIZE,
        'n_epochs': N_EPOCHS,
        'lr': LEARNING_RATE,
        'warmup': 0.2,
        'w_decay': 0.01,
        'threshold': THRESHOLD
    }

    model = MLTHSClassifier(config)

    trainer = pl.Trainer(logger=logger, max_epochs=config['n_epochs'], num_sanity_val_steps=1, enable_progress_bar=True, log_every_n_steps=10)
    trainer.fit(model, mlths_data_module)
    


    


        





