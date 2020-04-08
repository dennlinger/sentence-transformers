"""
The system RoBERTa trains on the AGB dataset  with softmax loss function.
At every 1000 training steps, the model is evaluated on the AGB dev set.
"""
from torch.utils.data import DataLoader
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, LabelAccuracyEvaluator
from sentence_transformers.readers import *
import logging
import torch
import math
import os
from datetime import datetime

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# # Read the dataset
model_name = "/data/daumiller/sentence-transformers/examples/output/training_agb_og_roberta-base-2020-03-26_10-53-54"
batch_size = 32
# model_name = "output/training_agb_roberta-base-nli-mean-tokens-2020-03-19_14-10-11_sections_1"
# batch_size = 46

agb_reader = AGBDataReader('datasets/AGB_og')
train_num_labels = agb_reader.get_num_labels()


# Use RoBERTa for mapping tokens to embeddings
model = SentenceTransformer(model_name)

# Don't train model, only train the loss layer, to retrieve results
model.train(False)
for param in model.parameters():
    param.requires_grad = False

logging.info("Read AGB train dataset")
train_data = SentencesDataset(agb_reader.get_examples('train.tsv'), model=model, shorten=True)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.SoftmaxLoss(model=model,
                                sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                num_labels=train_num_labels)

logging.info("Read AGB dev dataset")
dev_data = SentencesDataset(examples=agb_reader.get_examples('dev.tsv'), model=model, shorten=True)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model=train_loss)

# Configure the training
num_epochs = 2

warmup_steps = math.ceil(len(train_data) * num_epochs / batch_size * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

model_save_path = 'output/retrain_'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=10000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )

torch.save(train_loss, os.path.join(model_name, "classifier.pt"))

logging.info("Read AGB test dataset")
test_data = SentencesDataset(examples=agb_reader.get_examples('test.tsv'), model=model, shorten=True)
# test_data = SentencesDataset(examples=agb_reader.get_examples('test_raw.tsv'), model=model, shorten=True)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
evaluator = LabelAccuracyEvaluator(test_dataloader, softmax_model=train_loss)

model.evaluate(evaluator)
