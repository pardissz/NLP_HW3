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
        self.indication = features['indication']
        self.pharmacodynamics = features['pharmacodynamics']
        self.mechanism_of_action = features['mechanism_of_action']
        self.toxicity = features['toxicity']
        self.metabolism = features['metabolism']
        self.absorption = features['absorption']
        self.half_life = features['half_life']
        self.protein_binding = features['protein_binding']
        self.route_of_elimination = features['route_of_elimination']

    def getDrugfeatures(self):
        drug_dict = {
            "name": self.name,
            "description": self.description,
            "simple_description": self.simple_description,
            "clinical_description": self.clinical_description,
            "synonyms": self.synonyms,
            "classification_description": self.classification_description,
            "indication": self.indication,
            "pharmacodynamics": self.pharmacodynamics,
            "mechanism_of_action": self.mechanism_of_action,
            "toxicity": self.toxicity,
            "metabolism": self.metabolism,
            "absorption": self.absorption,
            "half_life": self.half_life,
            "protein_binding": self.protein_binding,
            "route_of_elimination": self.route_of_elimination
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
        drug_indication = ''
        drug_pharmacodynamics = ''
        drug_mechanism_of_action = ''
        drug_toxicity = ''
        drug_metabolism = ''
        drug_absorption = ''
        drug_half_life = ''
        drug_protein_binding = ''
        drug_route_of_elimination = ''

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
                classification = drug[idx]
                for idx2, subfeature in enumerate(classification):
                    if 'description' in str(subfeature):
                        drug_classification_description = classification[idx2].text if classification[idx2].text is not None else ''
            if 'indication' in str(feature):
                drug_indication = drug[idx].text if drug[idx].text is not None else ''
            if 'pharmacodynamics' in str(feature):
                drug_pharmacodynamics = drug[idx].text if drug[idx].text is not None else ''
            if 'mechanism-of-action' in str(feature):
                drug_mechanism_of_action = drug[idx].text if drug[idx].text is not None else ''
            if 'toxicity' in str(feature):
                drug_toxicity = drug[idx].text if drug[idx].text is not None else ''
            if 'metabolism' in str(feature):
                drug_metabolism = drug[idx].text if drug[idx].text is not None else ''
            if 'absorption' in str(feature):
                drug_absorption = drug[idx].text if drug[idx].text is not None else ''
            if 'half-life' in str(feature):
                drug_half_life = drug[idx].text if drug[idx].text is not None else ''
            if 'protein-binding' in str(feature):
                drug_protein_binding = drug[idx].text if drug[idx].text is not None else ''
            if 'route-of-elimination' in str(feature):
                drug_route_of_elimination = drug[idx].text if drug[idx].text is not None else ''

        drug_dict = {
            "name": drug_name,
            "description": drug_description,
            "simple_description": drug_simple_description,
            "clinical_description": drug_clinical_description,
            "synonyms": drug_synonyms,
            "classification_description": drug_classification_description,
            "indication": drug_indication,
            "pharmacodynamics": drug_pharmacodynamics,
            "mechanism_of_action": drug_mechanism_of_action,
            "toxicity": drug_toxicity,
            "metabolism": drug_metabolism,
            "absorption": drug_absorption,
            "half_life": drug_half_life,
            "protein_binding": drug_protein_binding,
            "route_of_elimination": drug_route_of_elimination
        }

        drug = Drug(drug_dict)
        drug_list.append(drug.getDrugfeatures())

    df = pd.DataFrame(drug_list)
    df.to_csv(f'output5_{i}.csv', index=False)

df = pd.concat([pd.read_csv(f'output5_{i}.csv') for i in range(n)])
df.to_csv('output5.csv', index=False)

for i in range(n):
    os.remove(f'output5_{i}.csv')


import pandas as pd
df=pd.read_csv("output5.csv")
