"""
Generates three numpy pickled files that contain the data from our og-paragraph datset.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import json
import os


def write_tensors_to_pickle(files, out_fn="train.pkl"):
    all_tensors = []
    for fn in files:
        fpath = os.path.join("/data/daumiller/tos-section-analyzer/tos-data-og", fn)
        with open(fpath) as f:
            tos = json.load(f)

        all_texts = [para["text"] for para in tos["level1_headings"]]

        # get tensors in list form
        results = model.encode(all_texts)

        results = np.stack(results, axis=0)

        all_tensors.append(results)

    with open(out_fn, "wb") as f:
        pickle.dump(all_tensors, f)


if __name__ == "__main__":

    model = SentenceTransformer("/data/salmasian/sentence_transformers/run1/training_agb_roberta-base-nli-mean-tokens-2020-04-07_18-41-13_og_consec_1")

    train_fraction = 0.8
    dev_fraction = 0.1
    test_fraction = 0.1

    files = sorted(os.listdir("/data/daumiller/tos-section-analyzer/tos-data-og"))
    np.random.seed(69120)
    file_order = np.random.choice(files, len(files), replace=False)
    train_files = file_order[:int(len(files) * train_fraction)]
    dev_files = file_order[int(len(files) * train_fraction): int(len(files) * (train_fraction + dev_fraction))]
    test_files = file_order[int(len(files) * (train_fraction + dev_fraction)):]

    write_tensors_to_pickle(train_files, "train.pkl")
    write_tensors_to_pickle(dev_files, "dev.pkl")
    write_tensors_to_pickle(test_files, "test.pkl")

