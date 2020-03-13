from . import InputExample
import csv
import gzip
import os


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
