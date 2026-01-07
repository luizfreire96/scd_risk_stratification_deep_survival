import numpy as np
import pandas as pd
import json

info_df = pd.read_csv('raw_data/subject-info.csv', delimiter=';')

with open("variaveis_artigo.json", "r") as f:
    data = json.load(f)

vars_sem_laudo = data["vars_artigo"]

info_df = info_df[list(info_df.columns[:5]) + vars_sem_laudo]

info_df = info_df.rename({"QRS > 120 ms ": "QRS > 120 ms"})

################################
# substituindo saida do estudo #
################################
info_df.loc[info_df['Exit of the study'].isna(), 'Exit of the study'] = 0
info_df['Exit of the study'] = info_df['Exit of the study'].astype(int)

###############################################
# filtrando apenas causas de morte relevantes #
###############################################
info_df["Exit of the study"] = np.where(info_df['Exit of the study'] != 3, 0, 1)
obj_cols = info_df.columns[info_df.dtypes == 'object'][1:]
info_df.loc[info_df['Age'] == '>89', 'Age'] = '89'
for col in obj_cols:
    info_df[col] = info_df[col].str.replace(',', '.').astype(float)

info_df = info_df.dropna()
    
info_df.to_parquet("treated_data/tabular_data/tabular_data_treated.parquet")
