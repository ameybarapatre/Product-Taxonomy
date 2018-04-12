import numpy as np
from keras.layers import Input, Embedding, LSTM, Dense , GRU ,  multiply , dot , Lambda, RepeatVector , merge , add , TimeDistributed
from keras.models import Model
import pickle
from keras import backend as K
import tensorflow as tf
import random


index_dict = pickle.load(open('./pkl/index_dict.p','rb'))
vocab_dim = 300
n_symbols = len(index_dict) + 1
embedding_weights = np.loadtxt("./Data/embedding.out")




embedding_layer = Embedding(output_dim=300, input_dim=n_symbols, trainable=False)
embedding_layer.build((None,)) # if you don't do this, the next step won't work
embedding_layer.set_weights([embedding_weights])

desc_input = Input( shape=(None,), dtype='float32', name='desc_input')
cat_input = Input(shape=(None,), dtype='float32', name='cat_input')

desc_in = embedding_layer(desc_input)

cat_in = embedding_layer(cat_input)

cat_out = GRU(300,return_sequences=False ,input_shape=(None,300))(cat_in)

output_attention_mul = Lambda(lambda x : tf.multiply(x,cat_out))(desc_in)

lstm_out = LSTM(9 , return_sequences=False)(output_attention_mul)

output = Dense(1, activation='sigmoid')(lstm_out)

model = Model(inputs = [desc_input , cat_input] , outputs = [output])

model.compile(optimizer='rmsprop', loss='binary_crossentropy' , metrics=['acc'])



train_categories = pickle.load(open('./pkl/categories.p','rb'))
train_descriptions = pickle.load(open('./pkl/descriptions.p','rb'))
train_labels = pickle.load(open('./pkl/labels.p','rb'))

l = np.arange(0 , len(train_descriptions))
np.random.shuffle(l)
count = 0

for z in range(0,30):
    print("Here:",count)
    for i in l:
        count+=1
        model.fit([np.array(train_descriptions[i])[np.newaxis, :] , np.array(train_categories[i])[np.newaxis, :]], [np.array([train_labels[i]])[:,np.newaxis]],
              epochs=1, batch_size=1 , shuffle=True)





