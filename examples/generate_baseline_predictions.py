"""
The system RoBERTa trains on the AGB dataset  with softmax loss function.
At every 1000 training steps, the model is evaluated on the AGB dev set.
"""
from torch.utils.data import DataLoader
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, LabelGenerationEvaluator
from sentence_transformers.readers import *
import logging
import torch
import os
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# # Read the dataset
root_dir = "/data/salmasian/baselines"
for i in range(1, 6):
    run_dir = os.path.join(root_dir, f"run{i}")
    for model_dir in sorted(os.listdir(run_dir)):
        curr_dir = os.path.join(run_dir, model_dir)
        # skip non-consecutive models
        if "og_" not in curr_dir:
            continue
        print(f"Working on model {model_dir}")
        # Delete when we re-evaluate...
        os.remove(os.path.join(curr_dir, "prediction_labels.csv"))
        os.remove(os.path.join(curr_dir, "prediction_results.csv"))
        model_save_path = curr_dir
        batch_size = 52
        agb_reader = TestAGBReader('datasets/og-test')
        train_num_labels = agb_reader.get_num_labels()

        model = SentenceTransformer(model_save_path, device="cuda:1")

        train_loss = losses.SoftmaxLoss(model=model,
                                        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                        num_labels=train_num_labels)
        train_loss.classifier = torch.load(os.path.join(model_save_path, "2_Softmax/pytorch_model.bin"))

        print("test")
        test_dir = "/data/daumiller/sentence-transformers/examples/datasets/og-test"
        for fn in sorted(os.listdir(test_dir)):
            examples = agb_reader.get_examples(fn)
            if not examples:
                continue
            if len(examples) == batch_size + 1:
                batch_size_used = batch_size - 3
            else:
                batch_size_used = batch_size
            test_data = SentencesDataset(examples=examples, model=model, shorten=True)
            test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size_used)
            evaluator = LabelGenerationEvaluator(test_dataloader, softmax_model=train_loss)
            model.evaluate(evaluator, model_save_path)
