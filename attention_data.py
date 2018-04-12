#import spacy
#nlp = spacy.load('en_core_web_lg')

import pandas as pd
data = pd.read_csv("./Data/processed.csv")

data['label'] = 1

data.to_csv('./Data/labeled_processed.csv')

output = pd.DataFrame(columns=['Category' ,'Description' ,'label'] , dtype = 'str')



categories = pd.unique(data['Category'])

for idx ,  row in data.iterrows():
        #output = output.append(row , ignore_index=True)
        for each in categories:
            if each != row['Category']:
                output = output.append({ 'Category': each ,'Description' :  row['Description'] , 'label' : 0} , ignore_index=True)

output.to_csv('./Data/synthetic.csv')
