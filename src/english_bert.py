
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertModel, BertTokenizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

df = pd.read_csv('200+_name_dataset_bert10.csv')

def preprocess_data(text):
    text = re.sub(r'([.,;])', r'\1 ', text)
    text = re.sub(r'\[([^\]]+)\]', r'\1', text)
    text = re.sub(r'\((.*?)\)', r'\1', text)
    text = text.translate(str.maketrans('', '', string.punctuation.replace('-', '')))
    text = re.sub(r'\b\d+\b', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word.lower() not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    new_words = []
    skip = False
    for i in range(len(words)):
        if skip:
            skip = False
            continue
        if i < len(words)-1 and len(words[i+1]) == 1:
            new_words.append(words[i]+'-'+words[i+1])
            skip = True
        else:
            new_words.append(words[i])
    return new_words

def get_bert_embeddings(texts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
        outputs = model(**inputs)
        embedding = outputs[0][0].mean(dim=0).detach().cpu().numpy()
        embedding = embedding.reshape(1, -1)
        embeddings.append(embedding)
    return embeddings

def convert_string_to_array(s):
    s = s.replace('\n', '').replace('[', '').replace(']', '')
    s = np.fromstring(s, sep=' ')
    return s.reshape(-1, 768)

df['bert_embeddings'] = df['bert_embeddings'].apply(convert_string_to_array)


def find_similar_drugs(drug_description, top_n=3):
    drug_description = preprocess_data(drug_description)
    drug_embedding = np.mean([get_bert_embeddings([word])[0] for word in drug_description], axis=0)
    similarities = cosine_similarity(drug_embedding, np.concatenate(df['bert_embeddings'].values))

    top_drugs_indices = similarities.argsort()[0][::-1][:top_n]

    return df.iloc[top_drugs_indices]['name']


drug_usage = "this drug is using Antigen identified as immune target and is presented by APCs to T cells along with other molecules that stimulate their growth and activation."
print(find_similar_drugs(drug_usage))
