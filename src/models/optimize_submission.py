import json
from itertools import combinations
from tqdm import tqdm
import pandas as pd
import multiprocessing

CLASSES = ['building', 'pervious surface', 'impervious surface', 'bare soil', 'water', 'coniferous', 'deciduous',
           'brushwood', 'vineyard', 'herbaceous vegetation', 'agricultural land', 'plowed land']

df = pd.read_csv('../../data/results.csv')
sample_list = list(df['model '])
list_combinations = list()
# change_type_cols = ['weight'] + CLASSES
# df[change_type_cols] = df[change_type_cols].infer_objects()

for change_type_col in CLASSES:
    df[change_type_col] = df[change_type_col].str.replace(',', '.', regex=True)
    df[change_type_col] = df[change_type_col].astype(float)

for n in range(4, 16):
    list_combinations += list(combinations(sample_list, n))


def process_combination(chunk, results_list):
    for models in tqdm(chunk):
        models = list(models)
        filtered_df = df.loc[df['model '].isin(models)]
        if filtered_df['weight'].sum() <= 14 * 60 + 52:
            score_max = filtered_df[CLASSES].max(axis=0).mean()
            score_mean = filtered_df[CLASSES].mean(axis=0).mean()

            result = {
                'models': models,
                'score_max': score_max,
                'score_mean': score_mean,
                'score': (score_max + score_mean) / 2
            }
            results_list.append(result)


if __name__ == '__main__':
    num_processes = multiprocessing.cpu_count()  # Number of processes based on CPU cores

    manager = multiprocessing.Manager()
    results_list = manager.list()

    print('Calculate chunk size')
    chunk_size = len(list_combinations) // num_processes

    processes = []
    for i in tqdm(range(num_processes)):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_processes - 1 else len(list_combinations)
        chunk = list_combinations[start_idx:end_idx]
        p = multiprocessing.Process(target=process_combination, args=(chunk, results_list))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print('Convert the shared list to a regular list')
    results = list(results_list)

    print('Sort the results list by score in descending order')
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    print('Save the sorted results list to a JSON file')
    with open('../../data/results.json', 'w') as json_file:
        json.dump(results[:10], json_file)
