from IPython.core.display import HTML
import numpy as np
import string

import torch
import torch.nn.functional as F

from olm.engine import Engine, weight_of_evidence, difference_of_log_probabilities, calculate_correlation
from olm import InputInstance, Config
from olm.visualization import visualize_relevances
from olm.occlusion.explainer import GradxInputExplainer, IntegrateGradExplainer

from utils import words_from_text

from tqdm import tqdm
from collections import defaultdict

from segtok.tokenizer import web_tokenizer, space_tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from typing import List, Tuple
import pandas as pd

from keras.preprocessing.sequence import pad_sequences
import os
import torch.nn as nn
from functools import partial

import csv


class OLM(nn.Module):
    def __init__(self, MODEL_NAME, batch_size, device):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
        self.device = device
        self.batch_size = batch_size

    def byte_pair_offsets(self, input_ids, tokenizer):
        def get_offsets(tokens, start_offset):
            offsets = [start_offset]
            for t_idx, token in enumerate(tokens, start_offset):
                if not token.startswith(" "):
                    continue
                offsets.append(t_idx)
            offsets.append(start_offset + len(tokens))
            return offsets
            
        tokens = [tokenizer.convert_tokens_to_string(t)
                    for t in tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)]
        tokens = [token for token in tokens if token != "<pad>"]
        tokens = tokens[1:-1]
        
        offsets = get_offsets(tokens, start_offset=1)
        
        return offsets


    def dataset_to_input_instances(self, dataset: List[Tuple[List[str], str]]) -> List[InputInstance]:
        input_instances = []
        for idx, (sent, _) in enumerate(dataset):
            # tokens = self.tokenizer.tokenize(sent)
            # tokens = tokens[:450]
            # text = ' '.join([x for x in tokens])
            # fine_text = text.replace(' ##', '')

            # # limit the number of tokens to 510 for bert (since only bert is used for language modeling, see lm_sampling.py)
            # sent = words_from_text(sent)
            # # print(sent)
            # # print(len(sent))
            # sent = ' '.join(sent)

            # tok = BertTokenizer.from_pretrained('bert-base-uncased')
            # tokens = tok.tokenize(sent)
            # tokens = tokens[:510]
            # text = ' '.join(tokens).replace(' ##', '').replace(' - ', '-').replace(" ' ","'")
            final_text = sent.split()
            # print(final_text)
            # print(len(final_text))
            instance = InputInstance(id_=idx, sent=final_text)
            input_instances.append(instance)

        # print(input_instances)
        return input_instances


    def get_labels(self, dataset: List[Tuple[List[str], List[str], str]]) -> List[str]:
        return [int(label) for _, label in dataset]

    def collate_tokens(self, values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        size = max(v.size(0) for v in values)
        res = values[0].new(len(values), size).fill_(pad_idx)
        
        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                assert src[-1] == eos_idx
                dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
        return res

    def encode_instance(self, input_instance):
        return self.tokenizer.encode(text=" ".join(input_instance.sent.tokens),
                                add_special_tokens=True,
                                max_length=512,
                                return_tensors="pt", 
                                truncation = True)[0]

    def predict(self, input_instances, model, tokenizer, device):
        if isinstance(input_instances, InputInstance):
            input_instances = [input_instances]
        
        
        input_ids = [self.encode_instance(instance) for instance in input_instances]
        attention_mask = [torch.ones_like(t) for t in input_ids]

        input_ids = self.collate_tokens(input_ids, pad_idx=1).to(self.device)
        attention_mask = self.collate_tokens(attention_mask, pad_idx=0).to(self.device)
        
        
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        # logits.detach().cpu().numpy()
        return F.softmax(logits, dim=-1)


    def train_and_run(self, dataset, recipe):
        input_instances = self.dataset_to_input_instances(dataset)
        labels = self.get_labels(dataset)
        batch_size = self.batch_size

        ncorrect, nsamples = 0, 0
        for i in tqdm(range(0, len(input_instances), batch_size), total=len(input_instances) // batch_size):
            batch_instances = input_instances[i: i + batch_size]
            with torch.no_grad():
                probs = self.predict(batch_instances, self.model, self.tokenizer, self.device)
                # print(probs)
                predictions = probs.argmax(dim=-1).cpu().numpy().tolist()
                #print(predictions)
                for batch_idx, instance in enumerate(batch_instances):
                    # the instance id is also the position in the list of labels
                    idx = instance.id
                    true_label = labels[idx]
                    pred_label = predictions[batch_idx]
                    ncorrect += int(true_label == pred_label)
                    nsamples += 1
        print('| Accuracy: ', float(ncorrect)/float(nsamples))
        
        if recipe == 'advolm':
            config_resample = Config.from_dict({
            "strategy": "bert_lm_sampling",
            "cuda_device": 0,
            "bert_model": "bert-base-uncased",
            "batch_size": 8,
            "n_samples": 30,
            "std": False,
            "verbose": False  
            })
        
        else:
            config_resample = Config.from_dict({
            "strategy": "bert_lm_sampling",
            "cuda_device": 0,
            "bert_model": "bert-base-uncased",
            "batch_size": 8,
            "n_samples": 30,
            "std": True,
            "verbose": False  
            })


        def batcher(batch_instances):
            true_label_indices = []
            probabilities = []
            with torch.no_grad():
                probs = self.predict(batch_instances, self.model, self.tokenizer, self.device).cpu().numpy().tolist()
                for batch_idx, instance in enumerate(batch_instances):
                    # the instance id is also the position in the list of labels
                    idx = instance.id
                    true_label_idx = labels[idx]
                    true_label_indices.append(true_label_idx)
                    probabilities.append(probs[batch_idx][true_label_idx])
        
            return probabilities


        resample_engine = Engine(config_resample, batcher)
        instance_idx = 0
        n = len(dataset)    # total number of samples

        res_candidate_instances, res_candidate_results = resample_engine.run(input_instances[instance_idx: instance_idx+n])
        labels_true = labels[instance_idx: instance_idx+n]
        labels_pred = [self.predict(instance, self.model, self.tokenizer, self.device)[0].argmax().item() for instance in input_instances[instance_idx: instance_idx+n]]

        res_relevances = resample_engine.relevances(res_candidate_instances, res_candidate_results)

        return res_relevances, input_instances, labels_true, labels_pred

