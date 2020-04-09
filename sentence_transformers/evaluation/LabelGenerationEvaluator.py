from . import SentenceEvaluator
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from ..util import batch_to_device
import numpy as np
import os
import csv


class LabelGenerationEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy on a labeled dataset.
    Also generate output based on the format needed for the AGB task.

    This requires a model with LossFunction.SOFTMAX

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, dataloader: DataLoader, name: str = "", softmax_model=None, device="cuda:1"):
        """
        Constructs an evaluator for the given dataset

        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.name = name
        self.softmax_model = softmax_model
        self.softmax_model.to(self.device)

        if name:
            name = "_"+name

        self.csv_file = "prediction"+name+"_results.csv"
        self.csv_labels = "prediction"+name+"_labels.csv"

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        total = 0
        correct = 0

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        preds = None
        out_label_ids = None
        logging.info("Evaluation on the "+self.name+" dataset"+out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        for step, batch in enumerate(tqdm(self.dataloader, desc="Evaluating")):
            features, label_ids = batch_to_device(batch, self.device)
            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)

            total += prediction.size(0)
            correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()

            if preds is None:
                preds = prediction.detach().cpu().numpy()
                out_label_ids = label_ids.detach().cpu().numpy()
            else:
                preds = np.append(preds, prediction.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, label_ids.detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)
        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            label_path = os.path.join(output_path, self.csv_labels)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([preds])
                with open(label_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([out_label_ids])
            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([preds])
                with open(label_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([out_label_ids])

        return correct / total
