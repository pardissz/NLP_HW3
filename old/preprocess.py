import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

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
    return ' '.join(new_words)



import pandas as pd

df = pd.read_csv('output5 - Copy.csv')
df = df.dropna(subset=['description'])

df['preprocessed_ALL'] = df.apply(lambda row: ' '.join([preprocess_data(str(row[col])) for col in df.columns if pd.notnull(row[col])]), axis=1)
df.to_csv('New_ONLYDES.csv', index=False)
#print(preprocess_data("TNK-tPA This Lepirudin is used as an anticoagulant in patients with heparin-induced thrombocytopenia (HIT), an immune reaction associated with a high risk of thromboembolic complications.[A3, L41539] HIT is caused by the expression of immunoglobulin G (IgG) antibodies that bind to the complex formed by heparin and platelet factor 4. This activates endothelial cells and platelets and enhances the formation of thrombi.[A246609] Bayer ceased the production of lepirudin (Refludan) effective May 31, 2012.[L41574][Leu1, Thr2]-63-desulfohirudin;Desulfatohirudin;Hirudin variant-1;Lepirudin;Lepirudin recombinant;R-hirudin Cetuximab,Cetuximab is a recombinant chimeric human/mouse IgG1 monoclonal antibody that competitively binds to epidermal growth factor receptor (EGFR) and competitively inhibits the binding of epidermal growth factor (EGF).[A227973] EGFR is a member of the ErbB family of receptor tyrosine kinases found in both normal and tumour cells; it is responsible for regulating epithelial tissue development and homeostasis.[A228083] EGFR has been implicated in various types of cancer, as it is often overexpressed in malignant cells [A227973] and EGFR overexpression has been linked to more advanced disease and poor prognosis.[A227963] EGFR is often mutated in certain types of cancer and serves as a driver of tumorigenesis.[A228083] _In vitro_, cetuximab was shown to mediate anti-tumour effects in numerous cancer cell lines and human tumour xenografts.[A227963]"))


import pandas as pd

df = pd.read_csv('New_ONLYDES.csv')

df[['name', 'preprocessed_ALL']].to_csv('MODIFIED_FINAL_DESC.csv', index=False)
