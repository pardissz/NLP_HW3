from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import numpy as np

from tqdm import tqdm
import numpy as np
import pandas as pd
"""

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

df = pd.read_csv('200+_name_dataset.csv')

def get_bert_embeddings(texts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
        outputs = model(**inputs)
        # Use the [CLS] token representation as sentence embedding
        embedding = outputs[0][:, 0, :].detach().cpu().numpy()
        embeddings.append(embedding)
    return embeddings

tqdm.pandas()
batch_size = 20  # Reduced batch size
bert_embeddings = []
for i in tqdm(range(0, len(df), batch_size)):
    bert_embeddings.extend(get_bert_embeddings(df['preprocessed_ALL'].iloc[i:i+batch_size]))

bert_embeddings = np.concatenate(bert_embeddings)

np.save('bertnew.npy', bert_embeddings)

"""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
bert_embeddings = np.load('bertnew.npy')

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

df = pd.read_csv('../data/200+_name_dataset.csv')

def get_bert_embeddings(texts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
        outputs = model(**inputs)
        embedding = outputs[0][:, 0, :].detach().cpu().numpy()
        embeddings.append(embedding)
    return embeddings
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
def find_similar_drugs(drug_description, top_n=3):
    drug_description = preprocess_data(drug_description)

    drug_embedding = get_bert_embeddings([drug_description])[0]

    similarities = cosine_similarity(drug_embedding, bert_embeddings)

    top_drugs_indices = similarities.argsort()[0][::-1][:top_n]

    return df.iloc[top_drugs_indices]['name']

text = "HIV infection prevention drug that virus HIV would vanish and needed skin infection be healed from fungas"
print(find_similar_drugs(text))
