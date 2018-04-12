import numpy as np
from keras.layers import Input, Embedding, LSTM, Dense , GRU ,  multiply , dot , Lambda, RepeatVector , merge , add
from keras.models import Model
import pickle
from keras import backend as K
import tensorflow as tf


index_dict = pickle.load(open('./pkl/index_dict.p','rb'))
vocab_dim = 300
n_symbols = len(index_dict) + 1
embedding_weights = np.loadtxt("./Data/embedding.out")

embedding_layer = Embedding(output_dim=300, input_dim=n_symbols, trainable=False)
embedding_layer.build((None,)) # if you don't do this, the next step won't work
embedding_layer.set_weights([embedding_weights])



desc_input = Input(shape=(None,300), dtype='float32', name='desc_input')
cat_input = Input(shape=(None,300), dtype='float32', name='cat_input')

desc_in = embedding_layer(desc_input)
cat_in = embedding_layer(cat_input)

time_step = K.int_shape(desc_input)[0]

cat_out = GRU(300,return_sequences=False)(cat_input)

#a = RepeatVector(time_step)(cat_out)

#output_attention_mul =  K.batch_dot(desc_in , cat_out)
#output_attention_mul = Lambda(lambda x : t.multiply(x, cat_out))(desc_in)
output_attention_mul = add([desc_in , cat_out])

lstm_out = LSTM(9 , return_sequences=False)(output_attention_mul)

output = Dense(1, activation='sigmoid')(lstm_out)

model = Model(inputs = [desc_input , cat_input] , output = [output])

model.compile(optimizer='rmsprop', loss='binary_crossentropy')
