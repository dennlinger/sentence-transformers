"""
The system RoBERTa trains on the AGB dataset  with softmax loss function.
At every 1000 training steps, the model is evaluated on the AGB dev set.
"""
import argparse

from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, LabelAccuracyEvaluator
from sentence_transformers.readers import *
import logging
from datetime import datetime
import torch
import os

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train a shallow network with tfidf for classification.')
    parser.add_argument('--num_epochs', help='training epochs',
                        type=int, default=2)
    parser.add_argument('--batch_size', help='training batch size',
                        type=int, default=32)
    parser.add_argument('--intermediate_eval_steps', help='run evaluation after how many steps.',
                        type=int, default=30000)
    parser.add_argument('--output_folder', help='The location for the output folder ',
                        default="output/")
    parser.add_argument('--dataset_folder', help='The location for the dataset ',
                        default="'datasets/AGB_og_consec'")


    args = parser.parse_args()
    # Read the dataset
    model_name = 'roberta-base-nli-mean-tokens'
    batch_size = args.batch_size
    agb_reader = AGBDataReader(args.dataset_folder)
    train_num_labels = agb_reader.get_num_labels()
    model_save_path = os.path.join(args.output_folder,'training_agb_'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    model = SentenceTransformer(model_name)

    # Convert the dataset to a DataLoader ready for training
    logging.info("Read AGB train dataset")
    train_data = SentencesDataset(agb_reader.get_examples('train_raw.tsv'), model=model, shorten=True)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    train_loss = losses.SoftmaxLoss(model=model,
                                    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                    num_labels=train_num_labels)

    logging.info("Read AGB dev dataset")
    dev_data = SentencesDataset(examples=agb_reader.get_examples('dev_raw.tsv'), model=model, shorten=True)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
    evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model=train_loss)

    # Configure the training
    num_epochs = args.num_epochs

    warmup_steps = math.ceil(len(train_data) * num_epochs / batch_size * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))


    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=args.intermediate_eval_steps,
              warmup_steps=warmup_steps,
              output_path=model_save_path
              )
    os.mkdir(os.path.join(model_save_path,"2_Softmax"))

    torch.save(train_loss.classifier,os.path.join(model_save_path,"2_Softmax/pytorch_model.bin"))


    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################

    model = SentenceTransformer(model_save_path)
    test_data = SentencesDataset(examples=agb_reader.get_examples('test_raw.tsv'), model=model, shorten=True)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    train_loss.classifier=torch.load(os.path.join(model_save_path,"2_Softmax/pytorch_model.bin"))
    evaluator = LabelAccuracyEvaluator(test_dataloader, softmax_model=train_loss)

    model.evaluate(evaluator)
