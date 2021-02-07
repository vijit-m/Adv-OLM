import pandas as pd
from transformers import BertTokenizer
import csv

def words_from_text(s, words_to_ignore=[]):
    """Lowercases a string, removes all non-alphanumeric characters, and splits
    into words."""
    # TODO implement w regex
    words = []
    word = ""
    for c in " ".join(s.split()):
        if c.isalnum():
            word += c
        elif c in "'-" and len(word) > 0:
            # Allow apostrophes and hyphens as long as they don't begin the
            # word.
            word += c
        elif word:
            if word not in words_to_ignore:
                words.append(word)
            word = ""
    if len(word) and (word not in words_to_ignore):
        words.append(word)
    return words


def read_imdb_dataset(path, data):
        df = pd.read_csv(path)
        dataset = []
        
        if data == '500samples_imdb.csv':
            s_list = df["sentence"]
            s_label = df["polarity"]
        elif data == '500samples_mnli.csv':
            s_list = df["sentence"]
            s_label = df["polarity"]
        else:
            s_list = df["text"]
            s_label = df["label"]

        for index, label in enumerate(s_label):
            sent = s_list[index]
            sent = sent.replace("<br />"," ")
            
            # limit the number of tokens to 510 for bert (since only bert is used for language modeling, see lm_sampling.py)
            sent = words_from_text(sent)
            # print(sent)
            # print(len(sent))
            sent = ' '.join(sent)

            tok = BertTokenizer.from_pretrained('bert-base-uncased')
            tokens = tok.tokenize(sent)
            tokens = tokens[:510]
            text = ' '.join(tokens).replace(' ##', '').replace(' - ', '-').replace(" ' ","'")

            target = int(label)
            dataset.append((text, target))
                
        return dataset

def write_to_csv(olm_file_name, res_relevances, input_instances, labels_true, labels_pred):
    f = open("olm-files/" + olm_file_name, "w")
    writer = csv.writer(f)  

    writer.writerow(['tokenized_text', 'relevances', 'true label', 'predicted label'])
    
    for i in range(len(res_relevances)):
        list_relevances = []
        for k in range(len(res_relevances[i])):
            list_relevances.append(res_relevances[i][('sent',k)])

        writer.writerow([input_instances[i].sent.tokens, list_relevances, labels_true[i], labels_pred[i]])           
    
    f.close()