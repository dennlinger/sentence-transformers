from . import InputExample
import json
import os


class TestAGBReader(object):
    """
    Reads in the Heidelberg AGB dataset
    """
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

    def get_examples(self, filename, max_examples=0):
        """

        :param filename:
        :param max_examples:
        :return:
        """
        with open(os.path.join(self.dataset_folder, filename)) as f:
            data = json.load(f)

        data = data["level1_heading"]
        examples = []
        prev_text = data[0]["text"]
        prev_label = data[0]["section"]
        if len(data) < 2:
            return None

        for i, paragraph in enumerate(data[1:]):
            guid = "%s-%s" % (filename, i)
            if paragraph["section"] == prev_label:
                temp_label = 1
            else:
                temp_label = 0

            examples.append(InputExample(guid=guid, texts=[prev_text, paragraph["text"]], label=temp_label))

            prev_text = paragraph["text"]
            prev_label = paragraph["section"]

        return examples

    @staticmethod
    def get_labels():
        # Adding different types of labels to assert correct conversion
        return {"same_section": 1, "other_section": 0, "1": 1, "0": 0, 1: 1, 0: 0}

    def get_num_labels(self):
        return len(self.get_labels())

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]
