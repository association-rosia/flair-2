import pandas as pd
from scipy.special import softmax
import numpy as np

df = pd.read_csv('../../data/raw/labels-statistics-12.csv')
df['Freq.-train (%)'] = df['Freq.-train (%)'].apply(lambda x: 1 / x)

class_weights = softmax(list(df['Freq.-train (%)']))
class_weights_bis = np.append(softmax(class_weights[:-1]), [0])

print(class_weights)
print(class_weights_bis)

