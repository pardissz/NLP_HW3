import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
import os
class Drug:
    def __init__(self, features):
        self.name = features['name']
        self.description = features['description']
        self.simple_description = features['simple_description']
        self.clinical_description = features['clinical_description']
        self.synonyms = features['synonyms']
        self.classification_description = features['classification_description']

    def getDrugfeatures(self):
        drug_dict = {
            "name": self.name,
            "description": self.description,
            "simple_description": self.simple_description,
            "clinical_description": self.clinical_description,
            "synonyms": self.synonyms,
            "classification_description": self.classification_description
        }
        return drug_dict

dB_file = '../full database.xml'
drugs = list(ET.parse(dB_file).getroot())
n = 50
len_chunk = len(drugs) // n

for i in tqdm(range(n)):
    chunk = drugs[i*len_chunk:(i+1)*len_chunk]
    drug_list = []

    for drug in chunk:
        drug_name = ''
        drug_description = ''
        drug_simple_description = ''
        drug_clinical_description = ''
        drug_synonyms = ''
        drug_classification_description = ''

        for idx, feature in enumerate(drug):
            if 'name' in str(feature):
                drug_name = drug[idx].text if drug[idx].text is not None else ''
            if 'description' in str(feature):
                drug_description = drug[idx].text if drug[idx].text is not None else ''
            if 'simple-description' in str(feature):
                drug_simple_description = drug[idx].text if drug[idx].text is not None else ''
            if 'clinical-description' in str(feature):
                drug_clinical_description = drug[idx].text if drug[idx].text is not None else ''
            if 'synonyms' in str(feature):
                drug_synonyms = ';'.join([synm.text for synm in list(drug[idx])]) if list(drug[idx]) else ''
            if 'classification' in str(feature):
                drug_classification_description = drug[idx].find('description').text if drug[idx].find('description') is not None else ''

        drug_dict = {
            "name": drug_name,
            "description": drug_description,
            "simple_description": drug_simple_description,
            "clinical_description": drug_clinical_description,
            "synonyms": drug_synonyms,
            "classification_description": drug_classification_description
        }

        drug = Drug(drug_dict)
        drug_list.append(drug.getDrugfeatures())

    df = pd.DataFrame(drug_list)
    df.to_csv(f'output_{i}.csv', index=False)

df = pd.concat([pd.read_csv(f'output_{i}.csv') for i in range(50)])
df.to_csv('output.csv', index=False)

for i in range(50):
    os.remove(f'output_{i}.csv')
