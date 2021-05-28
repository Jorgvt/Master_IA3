import random

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

from data_utils import SummarizerDataset, load_data
from trainer import SummarizerTrainer

# Data
DATA_TR_PATH = './data/news_summary.csv'
SEED = 42
TRAINING_RATIO = .8

# Model
MODEL_NAME = 't5-small'
MAX_LEN_SEQ = 512
MAX_SUMMARY_LEN = 150
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'

# Optimization parameters
N_EPOCHS = 2
BATCH_SIZE = 2
BATCH_SIZE_VAL = 2
LEARNING_RATE = 1e-4  # 2e-4
OPTIMIZER = Adam

# Seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Load data and split into training/test
training_set, val_set = load_data(path='./data/news_summary.csv', 
                                  tr_ratio=TRAINING_RATIO, 
                                  seed=SEED)

training_set = training_set[0:4]
val_set = val_set[0:4]

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
summarizer = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Create loaders for datasets
training_set = SummarizerDataset(dataset=training_set,
                                 tokenizer=tokenizer,
                                 article_len=MAX_LEN_SEQ,
                                 summary_len=MAX_SUMMARY_LEN)

val_set = SummarizerDataset(dataset=val_set,
                            tokenizer=tokenizer,
                            article_len=MAX_LEN_SEQ,
                            summary_len=MAX_SUMMARY_LEN)

dataloader_tr = DataLoader(dataset=training_set,
                           batch_size=BATCH_SIZE,
                           shuffle=True)

dataloader_val = DataLoader(dataset=val_set,
                            batch_size=BATCH_SIZE_VAL,
                            shuffle=False)

# Trainer
optimizer = OPTIMIZER(params=summarizer.parameters(), 
                      lr=LEARNING_RATE)
trainer = SummarizerTrainer(model=summarizer,
                            tokenizer=tokenizer,
                            optimizer=optimizer,
                            device=DEVICE,
                            n_epochs=N_EPOCHS)

# Train and validate model
trainer.train(dataloader_train=dataloader_tr,
              dataloader_val=dataloader_val)

# Use the model for inference (predictions)
paragraph = '''New Delhi, Apr 25 (PTI) Union minister Vijay Goel today batted for the unification of the three municipal corporations in the national capital saying a discussion over the issue was pertinent. The BJP leader, who was confident of a good show by his party in the MCD polls, the results of which will be declared tomorrow, said the civic bodies needed to be "revamped" in order to deliver the services to the people more effectively. The first thing needed was a discussion on the unification of the three municipal corporations and there should also be an end to the practice of sending Delhi government officials to serve in the civic bodies, said the Union Minister of State (Independent Charge) for Youth Affairs and Sports. "Barring one, the two other civic bodies have been incurring losses. It would be more fruitful and efficient if all the three were merged," he said, referring to the north, south and east Delhi municipal corporations. The erstwhile Municipal Corporation of Delhi (MCD) was trifurcated into NDMC, SDMC and EDMC by the then Sheila Dikshit-led Delhi government in 2012. Goel predicted a "thumping" victory for the BJP in the MCD polls. He said the newly-elected BJP councillors will be trained on the functioning of the civic bodies and dealing with the bureaucracy.'''
summary_pred = trainer.predict(article=paragraph, 
                               max_article_len=MAX_LEN_SEQ, 
                               max_summary_len=MAX_SUMMARY_LEN)

print(f'Paragraph: {paragraph}\n')
print(f'Predicted Summary: {summary_pred}')