import pandas as pd
import re
import nltk

df = pd.read_csv("goldstandard_eng_v1.csv" , sep =";" ,encoding = "latin-1")

clothing = df[df['GS1_Level1_Category'] == '67000000_Clothing']


data = pd.DataFrame(columns=['Category' ,'Description'] , dtype = 'str')


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


for each in pd.unique(clothing['GS1_Level3_Category']):
    l = re.sub('[^A-Za-z]+',' ', each )
    d= clothing[clothing['GS1_Level3_Category'] == each]
    d = d[d['s:description'].notnull()]

    for  index, row in d.iterrows():

        data = data.append({ 'Category': l ,  'Description' : remove_characters_before_tokenization(row['s:description'])} , ignore_index=True)

data.to_csv('full_clothing.csv')