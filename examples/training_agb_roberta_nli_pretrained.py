"""
The system RoBERTa trains on the AGB dataset  with softmax loss function.
At every 1000 training steps, the model is evaluated on the AGB dev set.
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, LabelAccuracyEvaluator
from sentence_transformers.readers import *
import logging
from datetime import datetime
import torch
import os

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset
model_name = 'roberta-base-nli-mean-tokens'
batch_size = 64
agb_reader = AGBDataReader('datasets/AGB')
train_num_labels = agb_reader.get_num_labels()
model_save_path = 'output/training_agb_'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

model = SentenceTransformer(model_name)


# Convert the dataset to a DataLoader ready for training
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


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=10000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )
os.mkdir(os.path.join(model_save_path,"2_Softmax"))

torch.save(train_loss.classifier,os.path.join(model_save_path,"2_Softmax/pytorch_model.bin"))


##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_data = SentencesDataset(examples=agb_reader.get_examples('test.tsv'), model=model, shorten=True)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
train_loss.classifier=torch.load(os.path.join(model_save_path,"2_Softmax/pytorch_model.bin"))
evaluator = LabelAccuracyEvaluator(test_dataloader, softmax_model=train_loss)

model.evaluate(evaluator)
