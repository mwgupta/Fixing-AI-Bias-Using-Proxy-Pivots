# import
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import pdb
import csv

path = 'data/adult.data'

column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'martial_status', 'occupation', 'relationship', 'race', 'sex',
                    'capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target']
input_data = (pd.read_csv(path, names=column_names,
                          na_values="?", sep=r'\s*,\s*', engine='python'))

# targets: 1 when someone makes over 50k , otherwise 0
y = (input_data['target'] == '>50K').astype(int)

# x featues: including protected classes
X = (input_data
     .drop(columns=['target', 'fnlwgt']) 
     .fillna('Unknown')
     .pipe(pd.get_dummies))

# z features:
Z_race = X[['race_Amer-Indian-Eskimo',
      'race_Asian-Pac-Islander',
      'race_Black',
      'race_Other',
      'race_White']]
Z_race = Z_race.rename(columns = {'race_Amer-Indian-Eskimo':0,
                          'race_Asian-Pac-Islander':1,
                          'race_Black':2,
                          'race_Other':3,
                          'race_White':4})

Z_sex = X[['sex_Female','sex_Male']]
Z_sex = Z_sex.rename(columns = {'sex_Female':0,'sex_Male':1})

X = X.drop(columns = ['race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other','race_White', 'sex_Female','sex_Male'])

n_clusters = 8
Kmean = KMeans(n_clusters=n_clusters)
Kmean.fit(X)

pdb.set_trace()

with open(path, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow([Kmean.labels_])

pdb.set_trace()

# read csv
input_data = pd.read_csv(path, skiprows = 1)

