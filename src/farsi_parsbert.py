import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased", model_max_length=512)
model = AutoModel.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")

farsi_drug_dataset = 'data/farsi_drug_data.csv'
exir_drug_dataset = 'data/exir_drugs.csv'
df_drug = pd.read_csv(farsi_drug_dataset)
df_exir = pd.read_csv(exir_drug_dataset)

def embed_description(desc):
    tokens = tokenizer(desc, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    model_embeddings = output.last_hidden_state.mean(dim=1).squeeze().numpy()
    return model_embeddings

result = []
for _, row in df_drug.iterrows():
    masraf_embedding = embed_description(row['mavared_masraf'])
    # name_embedding = embed_description(row['name_farsi'])
    # attention_embedding = embed_description(row['tavajohat'])
    result.append(masraf_embedding)

embeddings_drugbank = np.array(result)

result = []

for _, row in df_exir.iterrows():
    # name_embedding = embed_description(row['generic_name'])
    group_embedding = embed_description(row['goroh_darmani'])
    mavared_embedding = embed_description(row['mavared_masraf'])
    result.append(np.mean((group_embedding, mavared_embedding), axis=0))

embeddings_exir = np.array(result)
# embeddings_array = np.concatenate((embeddings_drugbank, embeddings_exir))

def recommend_drugs(input_description, top_k=3):
    input_embedding = embed_description(input_description)
    exir_similarities = cosine_similarity([input_embedding], embeddings_exir).flatten()
    exir_indices = exir_similarities.argsort()[-top_k:][::-1]
    recommended_drugs = df_exir.loc[exir_indices, 'generic_name'].tolist()
    drugbank_similarities = cosine_similarity([input_embedding], embeddings_drugbank).flatten()
    drugbank_indices = drugbank_similarities.argsort()[-top_k:][::-1]
    recommended_drugs.extend(df_drug.loc[drugbank_indices, 'name_farsi'].tolist())
    return recommended_drugs

# Example usage
drug_usage = "در درمان انواع سرطان کاربرد دارد و می‌تواند از پیشروی سرطان جلوگیری کند"
recommended_drugs = recommend_drugs(drug_usage)
print(f"Recommended drugs for '{drug_usage}': {recommended_drugs}")
