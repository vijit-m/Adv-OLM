import argparse

from adv_olm import AdvOLM
from olm_class import OLM
from console import Console
from utils import read_imdb_dataset
from utils import write_to_csv

import torch
import transformers
from textattack.models.tokenizers import AutoTokenizer
from textattack.models.wrappers import HuggingFaceModelWrapper
import textattack

import csv
import time
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='bert-base-uncased-imdb', help='bert| distilbert| roberta| albert, look at textattack models')
    # parser.add_argument('--dataset', default=None, help='Path to the dataset')
    # parser.add_argument('--wandb', action='store_true', default=False, help='do you want to enable weights and biases')
    parser.add_argument('--overwrite-olm', action='store_true', default=False, help='do you want to overwrite olm files')
    parser.add_argument('--recipe', default='advolm', help='advolm | advolms')
    parser.add_argument('--num-examples', type=int, default=500, help='how many examples to attack (put number <= 500)')
    parser.add_argument('--batch-size', type=int, default=8, help='to make it work on colab keep default batch size')
    args = parser.parse_args()
    print(args)
    
    model = args.model
    # dataset = args.dataset
    # wandb = args.wandb
    recipe = args.recipe
    num_examples = args.num_examples
    batch_size = args.batch_size
    overwrite = args.overwrite_olm

    testdir = './testdata/'
    MODEL_to_DATA = {
        "albert-base-v2-ag-news": '500samples_ag_news.csv',
        "albert-base-v2-imdb": '500samples_imdb.csv',
        "albert-base-v2-yelp-polarity": '500samples_yelp.csv',
        "bert-base-uncased-ag-news": '500samples_ag_news.csv',
        "bert-base-uncased-imdb": '500samples_imdb.csv',
        "bert-base-uncased-yelp-polarity": '500samples_yelp.csv',
        "distilbert-base-uncased-MNLI": '500samples_mnli.csv',
        "distilbert-base-uncased-imdb": '500samples_imdb.csv',
        "distilbert-base-uncased-ag-news": '500samples_ag_news.csv',
        "roberta-base-MNLI": '500samples_mnli.csv',
        "roberta-base-ag-news": '500samples_ag_news',
        "roberta-base-imdb": '500samples_imdb.csv',
    }   


    if recipe == 'advolm' or recipe == 'advolms':
        path_to_olm = 'olm-files/' + recipe + '-' + model + '-' + MODEL_to_DATA[model]
        MODEL_NAME = "textattack/" + model
        DATASET_PATH = "testdata/"
        path = os.path.join(DATASET_PATH, MODEL_to_DATA[model])
        dataset = read_imdb_dataset(path, MODEL_to_DATA[model])
        dataset = dataset[0:num_examples]

        if overwrite or not os.path.exists(path_to_olm):

            print("-----------> Calculate relevance <-----------------------")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            bs = batch_size
            olm = OLM(MODEL_NAME, bs, device)

            res_relevances, input_instances, labels_true, labels_pred = olm.train_and_run(dataset, recipe)

            olm_file_name = recipe + '-' + model + '-' + MODEL_to_DATA[model]
            write_to_csv(olm_file_name, res_relevances, input_instances, labels_true, labels_pred)

        rec = {'advolm': "AdvOLM", 'advolms': "AdvOLMs"}
        print(f"---------------------> {rec[recipe]} <---------------------------")
        attack = AdvOLM(model=model, path_to_olm=path_to_olm, path_to_examples='testdata/' + MODEL_to_DATA[model])

        result = Console(dataset, num_examples, attack)
        result.print_console()
