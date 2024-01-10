from hazm import word_tokenize, stopwords_list
import pandas as pd

farsi_drug_dataset = 'data/farsi_drug_data.csv'
exir_drug_dataset = 'data/exir_drugs.csv'
df_drug = pd.read_csv(farsi_drug_dataset)
df_exir = pd.read_csv(exir_drug_dataset)

stpwrds_list = stopwords_list()

def preprocess_farsi(text):
    characters_to_remove = ['"', ",", "/", "(", ")", " ها", " "]
    for char in characters_to_remove:
        text = text.replace(char, '')

    # Remove English characters
    text = ''.join([c for c in text if not (65 <= ord(c) <= 90 or 97 <= ord(c) <= 122)])

    words = word_tokenize(text)
    res = ""
    for word in words:
        if word not in stpwrds_list:
            res += word + " "

    return ' '.join(words)


result = []

for _, row in df_drug.iterrows():
    result.append({
        'name': preprocess_farsi(row['name_farsi']),
        'group': preprocess_farsi(row['goroh_darmani']),
        'usage': preprocess_farsi(row['mavared_masraf']),
    })

for _, row in df_exir.iterrows():
    result.append({
        'name': preprocess_farsi(row['generic_name']),
        'group': preprocess_farsi(row['goroh_darmani']),
        'usage': preprocess_farsi(row['mavared_masraf']),
    })

df = pd.DataFrame(result).dropna()
df.to_csv('combined_farsi_drug_dataset.csv', index=False, encoding='utf-8-sig')
