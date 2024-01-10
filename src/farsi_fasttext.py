from gensim.models import FastText
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from hazm import word_tokenize, stopwords_list

stop_words = set(stopwords_list())

df = pd.read_csv('combined_farsi_drug_dataset.csv')

df['preprocessed_ALL'] = df['name'] + " " + df['group'] + " " + df['usage']
df['preprocessed_ALL'] = df['preprocessed_ALL'].apply(lambda x: x.split())

ft_model = FastText(df['preprocessed_ALL'], vector_size=300, window=7, min_count=1, workers=4, sg=1)
ft_model.save("Farsi_fasttext_model")


def preprocess_farsi(text):
    characters_to_remove = ['"', ",", "/", "(", ")", " ها", " "]
    for char in characters_to_remove:
        text = text.replace(char, '')

    # Remove English characters
    text = ''.join([c for c in text if not (65 <= ord(c) <= 90 or 97 <= ord(c) <= 122)])

    words = word_tokenize(text)
    res = ""
    for word in words:
        if word not in stop_words:
            res += word + " "

    return ' '.join(words)

def get_fasttext_embeddings(words):
    embeddings = [ft_model.wv[word] for word in words if word in ft_model.wv.key_to_index]
    if not embeddings:
        return np.zeros((1, ft_model.get_dimension()))
    return np.mean(embeddings, axis=0).reshape(1, -1)

df['fasttext_embeddings'] = df['preprocessed_ALL'].apply(get_fasttext_embeddings)

def find_similar_drugs_fasttext(drug_description, top_n=3):
    drug_description = preprocess_farsi(drug_description)
    drug_embedding = get_fasttext_embeddings(drug_description)
    similarities = cosine_similarity(drug_embedding, np.concatenate(df['fasttext_embeddings'].values))

    top_drugs_indices = similarities.argsort()[0][::-1][:top_n]

    return df.iloc[top_drugs_indices]['name']

drug_usage = "سرطانی خونی سرطان پوست ملانوم بدخیم"
print(find_similar_drugs_fasttext(drug_usage))
