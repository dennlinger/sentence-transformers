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
import time
import os
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# # Read the dataset
# model_save_path = "/home/dennis/sentence-transformers/examples/output/training_agb_avg_word_embeddings-2020-03-27_17-23-51_og_consecutive_1"
# model_save_path = "/data/salmasian/baselines/run1/training_agb_avg_word_embeddings-2020-04-08_07-36-04_og_consec_1"
model_save_path = "/data/daumiller/sentence-transformers/examples/training_agb_roberta-base-2020-04-07_18-44-07_og_consec_1"

batch_size = 24
agb_reader = TestAGBReader('datasets/og-test')
train_num_labels = agb_reader.get_num_labels()

model = SentenceTransformer(model_save_path, device="cuda:1")

train_loss = losses.SoftmaxLoss(model=model,
                                sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                num_labels=train_num_labels)
train_loss.classifier = torch.load(os.path.join(model_save_path, "2_Softmax/pytorch_model.bin"))

print("test")
test_dir = "/data/daumiller/sentence-transformers/examples/datasets/og-test"
i = 100
start_time = time.time()
for fn in sorted(os.listdir(test_dir)):
    i += 1

    if i >= 100:
        break
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

end_time = time.time()
logging.info(f"Evaluation of 100 files took {end_time-start_time:.2f} seconds per file.")