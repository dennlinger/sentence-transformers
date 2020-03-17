"""
Simple annotation script that compares the results on the 100 first samples of the AGB dev set,
to judge human performance.
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from sentence_transformers.readers import *
import logging
from collections import namedtuple
from datetime import datetime
from typing import Union, List
import numpy as np
import csv
import sys
import os

csv.field_size_limit(sys.maxsize)


class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str, texts: List[str], label: Union[int, float]):
        """
        Creates one InputExample with the given texts, guid and label

        str.strip() is called on both texts.

        :param guid
            id for the example
        :param texts
            the texts for the example
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = [text.strip() for text in texts]
        self.label = label

class AGBDataReader(object):
    """
    Reads in the Heidelberg AGB dataset
    """
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

    def get_examples(self, filename, max_examples=0):
        """
        data_splits specified which data split to use (train, dev, test).
        Expects that self.dataset_folder contains the files in tsv (tab-separated form),
        with three columns (s1 \t s2 \t [0|1]
        """
        with open(os.path.join(self.dataset_folder, filename)) as f:
            rows = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)

            examples = []
            id = 0
            for sentence_a, sentence_b, label in rows:
                guid = "%s-%d" % (filename, id)
                id += 1
                examples.append(InputExample(guid=guid, texts=[sentence_a, sentence_b], label=self.map_label(label)))

                if 0 < max_examples <= len(examples):
                    break

        return examples

    @staticmethod
    def get_labels():
        # Adding different types of labels to assert correct conversion
        return {"same_section": 1, "other_section": 0, "1": 1, "0": 0, 1: 1, 0: 0}

    def get_num_labels(self):
        return len(self.get_labels())

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]



# Read the dataset
agb_reader = AGBDataReader('datasets/AGB')
train_num_labels = agb_reader.get_num_labels()
model_save_path = 'output/training_agb_'+'human-baseline'+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.info("Read AGB dev dataset")
samples = agb_reader.get_examples('dev.tsv')

# start sample if we want to skip a certain number
start_sample = 0
# Number of samples to evaluate
num_samples = 100
np.random.seed(12)
indices = np.random.choice(range(len(samples)), len(samples), replace=False)

indices = indices[start_sample:start_sample+num_samples]

##############################################################################
#
# Load the dev samples and ask for human evaluation after output.
#
##############################################################################
answers = []
answertuple = namedtuple('user', 'ground truth')
for idx in indices:
    curr_sample = samples[idx]
    print("Sample 1:")
    print(curr_sample.texts[0])
    print("\nSample 2:")
    print(curr_sample.texts[1])
    answer = int(input("Are the two sections covering the same topic? (0 no, 1 yes)"))
    answers.append(answertuple(answer, curr_sample.label))
    print("\n\n\n")

os.makedirs(model_save_path, exist_ok=True)
with open(os.path.join(model_save_path, "judgements.txt"), "w") as f:
    for answer in answers:
        f.write(f"{answer[0]}\t{answer[1]}\n")
