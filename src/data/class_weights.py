import numpy as np
import pandas as pd
from scipy.special import softmax

df = pd.read_csv('../../data/raw/labels-statistics-12.csv')
# df['Freq.-train (%)'] = df['Freq.-train (%)'].apply(lambda x: 1 / x)
df['Freq.-test (%)'] = df['Freq.-test (%)'].apply(lambda x: 1 / x)

class_weights = softmax(list(df['Freq.-test (%)']))
class_weights_bis = np.append(softmax(list(df['Freq.-test (%)'])[:-1]), [0])

print(list(class_weights))
print(list(class_weights_bis))