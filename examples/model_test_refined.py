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
import os
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout    output2/run1/training_agb_roberta-base-2020-04-07_18-44-28_sections_1/

# # Read the dataset
model_save_path = "output2/run5/training_agb_bow-2020-04-08_23-10-14_og_consec_5"
batch_size = 52
agb_reader = AGBDataReader('datasets/AGB')
train_num_labels = agb_reader.get_num_labels()


# Use RoBERTa for mapping tokens to embeddings
model = SentenceTransformer(model_save_path)

train_loss = losses.SoftmaxLoss(model=model,
                                sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                num_labels=train_num_labels)
train_loss.classifier=torch.load(os.path.join(model_save_path,"2_Softmax/pytorch_model.bin"))

print("dev")
test_data = SentencesDataset(examples=agb_reader.get_examples('dev_raw.tsv'), model=model, shorten=True)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
evaluator = LabelAccuracyEvaluator(test_dataloader, softmax_model=train_loss)
model.evaluate(evaluator)
print("test")
test_data = SentencesDataset(examples=agb_reader.get_examples('test_raw.tsv'), model=model, shorten=True)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
evaluator = LabelAccuracyEvaluator(test_dataloader, softmax_model=train_loss)
model.evaluate(evaluator)