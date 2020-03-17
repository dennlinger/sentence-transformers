"""
This example uses a simple bag-of-words (BoW) approach. A sentence is mapped
to a sparse vector with e.g. 25,000 dimensions. Optionally, you can also use tf-idf.

To make the model trainable, we add multiple dense layers to create a Deep Averaging Network (DAN).
"""
import json
import logging
import math
import os
import re
from collections import Counter
from datetime import datetime

import nltk
from torch.utils.data import DataLoader
from tqdm import tqdm

from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers import models, losses
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from sentence_transformers.models.tokenizer.WordTokenizer import ENGLISH_STOP_WORDS
from sentence_transformers.readers import *

# nltk.download('punkt')
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset
batch_size = 32
agb_reader = AGBDataReader('datasets/AGB')
model_save_path = 'output/training_agb_bow-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
create_word_frequencies = True  # if set to true it will generate the term frequency file from all the agbs
train_num_labels = agb_reader.get_num_labels()

# Create the vocab for the BoW model
stop_words = ENGLISH_STOP_WORDS
max_vocab_size = 25000  # This is also the size of the BoW sentence vector.

# create a file with the word frequncies
if create_word_frequencies:
    folder = "/home/salmasian/tos-section-analyzer/tos-data-cleaned"
    counter = Counter()
    all_words = []
    for f in tqdm(sorted(os.listdir(folder))):
        with open(os.path.join(folder, f)) as fp:
            tos = json.load(fp)

        for section in tos["level1_headings"]:
            text = re.sub("</?[^>]*>", "", section["text"].strip())

            all_words = all_words + [word.lower() for word in nltk.tokenize.word_tokenize(text) if
                                     len(word) > 1 and not word.isnumeric()]
            # counter.update(w.lower().rstrip(punctuation) for w in text.split(' ') if len(w)>0 and w[0] != "(")
    all_word_freq = nltk.FreqDist(all_words)
    with open("datasets/extras/agb_doc_frequencies.txt", "w") as f:
        f.write(str(len(os.listdir(folder))) + "\n")
        for word, count in all_word_freq.most_common():
            if len(word) > 1:
                f.write(f"{word}\t{count}\n")

# Read the most common max_vocab_size words. Skip stop-words
vocab = set()
weights = {}
lines = open('datasets/extras/agb_doc_frequencies.txt').readlines()
num_docs = int(lines[0])
for line in lines[1:]:
    print(line.lower().strip().split("\t"))
    word, freq = line.lower().strip().split("\t")
    if word in stop_words:
        continue

    vocab.add(word)
    weights[word] = math.log(num_docs / int(freq))

    if len(vocab) >= max_vocab_size:
        break

# Create the BoW model. Because we set word_weights to the IDF values and cumulative_term_frequency=True, we
# get tf-idf vectors. Set word_weights to an empty dict and cumulative_term_frequency=False to get a 1-hot sentence encoding
bow = models.BoW(vocab=vocab, word_weights=weights, cumulative_term_frequency=True)

# Add two trainable feed-forward networks (DAN) with max_vocab_size -> 768 -> 512 dimensions.
sent_embeddings_dimension = max_vocab_size
dan1 = models.Dense(in_features=sent_embeddings_dimension, out_features=768)
dan2 = models.Dense(in_features=768, out_features=512)

model = SentenceTransformer(modules=[bow, dan1, dan2])

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

##############################################################################
#
# Load the stored model and evaluate its performance on AGB dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_data = SentencesDataset(examples=agb_reader.get_examples('test_raw.tsv'), model=model, shorten=False)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
evaluator = LabelAccuracyEvaluator(test_dataloader, softmax_model=train_loss)

model.evaluate(evaluator)
