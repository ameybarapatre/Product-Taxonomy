import nltk
import pandas as pd
import re


df = pd.read_csv('clothing.csv')


def remove_characters_before_tokenization(sentence,
                                          keep_apostrophes=False):
    sentence = sentence.strip()

    PATTERN = r'<[^>]*>'
    filtered_sentence = re.sub(PATTERN, r'', sentence)
    PATTERN = r'[?|$|&|*|%|@|(|)|~|x]'
    filtered_sentence = re.sub(PATTERN, r'', filtered_sentence)
    PATTERN = r'[^a-zA-Z0-9 ]'
    filtered_sentence = re.sub(PATTERN, r'', filtered_sentence)

    return filtered_sentence



data = pd.DataFrame(columns=['Category' ,'Description'] , dtype = 'str')

for index, row in df.iterrows():

        desc = remove_characters_before_tokenization(row['Description'])

        data = data.append( { 'Category' :  row['Category'] , 'Description': desc}, ignore_index=True)


data.to_csv("processed.csv")
