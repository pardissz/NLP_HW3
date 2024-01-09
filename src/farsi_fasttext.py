import re
import string

import fasttext
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from hazm import Normalizer, word_tokenize, Lemmatizer, stopwords_list

stop_words = set(stopwords_list())

fasttext_model_path = 'cc.fa.300.bin'
ft_model = fasttext.load_model(fasttext_model_path)

normalizer = Normalizer()
lemmatizer = Lemmatizer()

df = pd.read_csv('farsi_drug_data.csv')

def preprocess_data(text):
    text = normalizer.normalize(text)
    text = re.sub(r'([.,;،؛])', r'\1 ', text)
    text = re.sub(r'\[([^\]]+)\]', r'\1', text)
    text = re.sub(r'\((.*?)\)', r'\1', text)
    text = text.translate(str.maketrans('', '', string.punctuation.replace('-', '')))
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

def get_fasttext_embeddings(words):
    embeddings = [ft_model[word] for word in words if word in ft_model]
    if not embeddings:
        return np.zeros((1, ft_model.get_dimension()))
    return np.mean(embeddings, axis=0).reshape(1, -1)

df['fasttext_embeddings'] = df['mavared_masraf'].apply(preprocess_data).apply(get_fasttext_embeddings)

def find_similar_drugs_fasttext(drug_description, top_n=3):
    drug_description = preprocess_data(drug_description)
    drug_embedding = get_fasttext_embeddings(drug_description)
    similarities = cosine_similarity(drug_embedding, np.concatenate(df['fasttext_embeddings'].values))

    top_drugs_indices = similarities.argsort()[0][::-1][:top_n]

    return df.iloc[top_drugs_indices]['name_farsi']

drug_usage = "این دارو برای آنتیژن شناخته شده به عنوان هدف ایمنی و ارائه شده توسط APC به T سل‌ها به همراه دیگر مولکول‌ها که رشد و فعال‌سازی آن‌ها را تحریک می‌کند، استفاده می‌شود."
print(find_similar_drugs_fasttext(drug_usage))
