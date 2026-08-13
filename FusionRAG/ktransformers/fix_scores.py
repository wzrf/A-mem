import csv
import os
import glob
import pandas as pd
import re
from rouge import Rouge
import string
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))
def _rouge1_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        # scores = rouge.get_scores(prediction, ground_truth, avg=True)
        scores = rouge.get_scores(normalize_answer(prediction), normalize_answer(ground_truth), avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-1"]["f"]
data_name = 'hotpotqa-200'

model_name = 'Llama-3.1-8B-Instruct'
file_path = f'../processCache/{data_name}/{model_name}'
directory = os.path.abspath(file_path)
number = int( data_name.split('-')[1])

csv_files = glob.glob(os.path.join(directory, '*.csv'))
txt_files = glob.glob(os.path.join(directory, '*.txt'))
for csv_file in csv_files:
    file = pd.read_csv(csv_file)
    if number != file.shape[0]:
        print(f'{csv_file} not complete')
    else:
        scores = 0
        for i,item in file.iterrows():

            scores += _rouge1_score(item['Pred Answer'], item['Real Answer'])
        print(csv_file, scores/number)