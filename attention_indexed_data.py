import pandas as pd
import pickle
from keras.preprocessing.text import Tokenizer
import numpy as np

tokenizer = pickle.load( open( "./pkl/tokenizer.p", "rb" ) )

index_dict = pickle.load( open( "./pkl/index_dict.p", "rb" ) )

word_vectors = pickle.load( open( "./pkl/word_vectors.p", "rb" ) )

#print(tokenizer.texts_to_sequences(['Upper Body Wear Tops']))


data = pd.read_csv("./Data/balanced.csv")


category = []
description = []
labels = []

for idx , row  in  data.iterrows():
    category.append(tokenizer.texts_to_sequences([row['Category']])[0])
    description.append( tokenizer.texts_to_sequences([row['Description']])[0])
    labels.append( row['label'])

pickle.dump( category, open( "./pkl/categories.p", "wb" ))
pickle.dump( description , open( "./pkl/descriptions.p", "wb" ))
pickle.dump( labels , open( "./pkl/labels.p", "wb" ))


num_words = len(index_dict.keys()) + 1
embedding_matrix = np.zeros((num_words, 300))
for word, i in index_dict.items():
    embedding_vector = word_vectors.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

np.savetxt("./Data/embedding.out",embedding_matrix)






