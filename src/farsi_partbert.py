import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# Load the ParsBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
model = AutoModel.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")

# Load the dataset
dataset_path = 'data/farsi_drug_data.csv'
df = pd.read_csv(dataset_path)
print(df['mavared_masraf'])

# Tokenize and embed the descriptions using ParsBERT
def embed_description(desc):
    tokens = tokenizer(desc, return_tensors='pt')
    with torch.no_grad():
        output = model(**tokens)
    model_embeddings = output.last_hidden_state.mean(dim=1).squeeze().numpy()
    return model_embeddings

# Create embeddings for all descriptions in the dataset
embeddings = []
for description in df['mavared_masraf']:
    embeddings.append(embed_description(description))

# Convert the embeddings list to a numpy array
embeddings_array = np.array(embeddings)

# Function to recommend drugs based on cosine similarity
def recommend_drugs(input_description, top_k=3):
    input_embedding = embed_description(input_description)
    similarities = cosine_similarity([input_embedding], embeddings_array).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    recommended_drugs = df.loc[top_indices, 'name_farsi'].tolist()
    return recommended_drugs

# Example usage
drug_usage = "دارویی برای تسکین درد بدن"
recommended_drugs = recommend_drugs(drug_usage)
print(f"Recommended drugs for '{drug_usage}': {recommended_drugs}")
