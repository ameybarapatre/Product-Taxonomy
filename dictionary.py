import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
import pickle

data = pd.read_csv('./Data/processed.csv')


texts = data['Category'] + ' ' +data['Description']




tokenizer = Tokenizer()

tokenizer.fit_on_texts(texts)

pickle.dump(tokenizer ,open("./pkl/tokenizer.p" , "wb"))

index_dict = tokenizer.word_index

pickle.dump(index_dict , open( "./pkl/index_dict.p", "wb" ))

word_vectors = dict.fromkeys(index_dict.keys())


import spacy
nlp = spacy.load('en_core_web_lg')

for each in word_vectors.keys():
    word_vectors[each] = nlp(each).vector

pickle.dump(word_vectors ,open( "./pkl/word_vectors.p", "wb" ))