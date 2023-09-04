import pandas as pd

df = pd.read_csv('../../data/raw/labels-statistics-12.csv')
df['Freq.-train (%)'] = df['Freq.-train (%)'].apply(lambda x: 1 / x)

print(list(df['Freq.-train (%)']))
