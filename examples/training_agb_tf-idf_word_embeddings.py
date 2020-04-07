"""
This example weights word embeddings (like GloVe) with IDF weights. The IDF weights can for example be computed on Wikipedia.

If 'glove.6B.300d.txt.gz' does not exist, it tries to download it from our server.

See https://public.ukp.informatik.tu-darmstadt.de/reimers/embeddings/ for available word embeddings files

You can get term-document frequencies from here:
https://public.ukp.informatik.tu-darmstadt.de/reimers/embeddings/wikipedia_doc_frequencies.txt
"""
import logging
import math
from datetime import datetime

from torch.utils.data import DataLoader

from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers import models, losses
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from sentence_transformers.readers import *
import os
import torch
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset
batch_size = 32
agb_reader = AGBDataReader('datasets/AGB_og_consec')
model_save_path = 'output/training_tf-idf_word_embeddings-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# os.environ["CUDA_VISIBLE_DEVICES"]="1"# for local testing
train_num_labels = agb_reader.get_num_labels()

# Map tokens to traditional word embeddings like GloVe
word_embedding_model = models.WordEmbeddings.from_text_file('datasets/extras/glove.6B.300d.txt.gz')

# Weight word embeddings using Inverse-Document-Frequency (IDF) values.
# For each word in the vocab ob the tokenizer, we must specify a weight value.
# The word embedding is then multiplied by this value
vocab = word_embedding_model.tokenizer.get_vocab()
word_weights = {}
lines = open('datasets/extras/agb_doc_frequencies.txt').readlines()
num_docs = int(lines[0])
for line in lines[1:]:
    word, freq = line.strip().split("\t")
    word_weights[word] = math.log(num_docs / int(freq))

# Words in the vocab that are not in the doc_frequencies file get a frequency of 1
unknown_word_weight = math.log(num_docs / 1)

# Initialize the WordWeights model. This model must be between the WordEmbeddings and the Pooling model
word_weights = models.WordWeights(vocab=vocab, word_weights=word_weights, unknown_word_weight=unknown_word_weight)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

# Add two trainable feed-forward networks (DAN)
sent_embeddings_dimension = pooling_model.get_sentence_embedding_dimension()
dan1 = models.Dense(in_features=sent_embeddings_dimension, out_features=sent_embeddings_dimension)
dan2 = models.Dense(in_features=sent_embeddings_dimension, out_features=sent_embeddings_dimension)

model = SentenceTransformer(modules=[word_embedding_model, word_weights, pooling_model, dan1, dan2])

# Convert the dataset to a DataLoader ready for training
logging.info("Read AGB train dataset")
train_data = SentencesDataset(agb_reader.get_examples('train_raw.tsv'), model=model, shorten=False)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.SoftmaxLoss(model=model,
                                sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                num_labels=train_num_labels)

logging.info("Read AGB dev dataset")
dev_data = SentencesDataset(examples=agb_reader.get_examples('dev_raw.tsv'), model=model, shorten=False)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model=train_loss)

# Configure the training
num_epochs = 10
warmup_steps = math.ceil(len(train_data) * num_epochs / batch_size * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )

os.mkdir(os.path.join(model_save_path,"2_Softmax"))
torch.save(train_loss.classifier,os.path.join(model_save_path,"2_Softmax/pytorch_model.bin"))

##############################################################################
#
# Load the stored model and evaluate its performance on AGB dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_data = SentencesDataset(examples=agb_reader.get_examples('test_raw.tsv'), model=model, shorten=False)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
train_loss.classifier=torch.load(os.path.join(model_save_path,"2_Softmax/pytorch_model.bin"))
evaluator = LabelAccuracyEvaluator(test_dataloader, softmax_model=train_loss)

model.evaluate(evaluator)
