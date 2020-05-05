"""
This example uses average word embeddings (for example from GloVe). It adds two fully-connected feed-forward layers (dense layers) to create a Deep Averaging Network (DAN).

If 'glove.6B.300d.txt.gz' does not exist, it tries to download it from our server.

See https://public.ukp.informatik.tu-darmstadt.de/reimers/embeddings/
for available word embeddings files
"""
import logging
import math
from datetime import datetime
from torch.utils.data import DataLoader
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers import models, losses
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from sentence_transformers.readers import AGBDataReader
import os
import torch
import argparse

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train a shallow network with avg of glove embeddings for classificagtion.')
    parser.add_argument('--num_epochs', help='training epochs',
                        type=int, default=10)
    parser.add_argument('--batch_size', help='training batch size',
                        type=int, default=32)
    parser.add_argument('--output_folder', help='The location for the output folder ',
                        default="output/")
    parser.add_argument('--dataset_folder', help='The location for the dataset ',
                        default="'datasets/AGB_og_consec'")
    parser.add_argument('--path_to_glove_model',
                        help='The location for the glove model word embeddings. ',
                        default="datasets/extras/glove.6B.300d.txt.gz")

    args = parser.parse_args()

    # Read the dataset
    batch_size = args.batch_size
    agb_reader = AGBDataReader(args.dataset_folder)
    model_save_path = os.path.join(args.output_folder,'training_agb_avg_word_embeddings-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    train_num_labels = agb_reader.get_num_labels()

    # Map tokens to traditional word embeddings like GloVe
    word_embedding_model = models.WordEmbeddings.from_text_file(args.path_to_glove_model)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    # Add two trainable feed-forward networks (DAN)
    sent_embeddings_dimension = pooling_model.get_sentence_embedding_dimension()
    dan1 = models.Dense(in_features=sent_embeddings_dimension, out_features=sent_embeddings_dimension)
    dan2 = models.Dense(in_features=sent_embeddings_dimension, out_features=sent_embeddings_dimension)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dan1, dan2])

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
    num_epochs = args.num_epochs
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
