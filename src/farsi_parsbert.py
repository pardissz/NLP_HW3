import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased", model_max_length=512)
model = AutoModel.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")

drug_dataset = 'combined_farsi_drug_dataset.csv'
df = pd.read_csv(drug_dataset, lineterminator='\n')

def embed_description(desc):
    tokens = tokenizer(desc, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    model_embeddings = output.last_hidden_state.mean(dim=1).squeeze().numpy()
    return model_embeddings

result = []
for _, row in df.iterrows():
    name_embedding = embed_description(row['name'])
    group_embedding = embed_description(row['group'])
    usage_embedding = embed_description(row['usage'])
    result.append(np.mean((name_embedding, group_embedding, usage_embedding), axis=0))

embeddings_array = np.array(result)

def recommend_drugs(input_description, top_k=3):
    input_embedding = embed_description(input_description)
    similarities = cosine_similarity([input_embedding], embeddings_array).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    recommended_drugs = df.loc[top_indices, 'name'].tolist()
    return recommended_drugs

# Example usage
drug_usage = "در درمان انواع سرطان کاربرد دارد و می‌تواند از پیشروی سرطان جلوگیری کند"
recommended_drugs = recommend_drugs(drug_usage)
print(f"Recommended drugs for '{drug_usage}': {recommended_drugs}")
