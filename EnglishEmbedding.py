import pandas as pd
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import numpy as np

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

df = pd.read_csv('200+_name_dataset.csv')
df['preprocessed_ALL'] = df['preprocessed_ALL'].apply(lambda x: x.split())

model = FastText(df['preprocessed_ALL'], vector_size=300, window=7, min_count=1, workers=4, sg=1)
model.save("English_fasttext_model")

model = FastText.load('English_fasttext_model')

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
words = list(model.wv.key_to_index.keys())
vectors = model.wv.vectors
word_vector_df = pd.DataFrame(vectors, index=words)
def find_similar_drugs(drug_description, model, top_n=3):
    drug_description = preprocess_data(drug_description)
    drug_embedding = np.mean([model.wv[word] for word in drug_description if word in model.wv.key_to_index], axis=0)
    drug_embeddings = df['preprocessed_ALL'].apply(lambda x: np.mean([model.wv[word] for word in x if word in model.wv.key_to_index], axis=0))

    similarities = cosine_similarity([drug_embedding], list(drug_embeddings))

    top_drugs_indices = similarities.argsort()[0][::-1][:top_n]

    return df.iloc[top_drugs_indices]['name']

text = "a drug which is good for cancer and helps when cancer is in its earlier stage"
print(find_similar_drugs(text, model))
