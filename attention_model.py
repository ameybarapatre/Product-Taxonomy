import numpy as np
from keras.layers import Input, Embedding, LSTM, Dense , GRU ,   Lambda, Dropout , Bidirectional
from keras.models import Model
import pickle
from keras import backend as K
import tensorflow as tf
import random
from keras.regularizers import l1_l2


index_dict = pickle.load(open('./pkl/index_dict.p','rb'))
vocab_dim = 300
n_symbols = len(index_dict) + 1
embedding_weights = np.loadtxt("./Data/embedding.out")




embedding_layer = Embedding(output_dim=300, input_dim=n_symbols, trainable=True)
embedding_layer.build((None,)) # if you don't do this, the next step won't work
embedding_layer.set_weights([embedding_weights])

desc_input = Input( shape=(None,), dtype='float32', name='desc_input')
cat_inp = Input(shape=(None,), dtype='float32', name='cat_input')

desc_in = embedding_layer(desc_input)

cat_in = embedding_layer(cat_inp)

cat_drop = Dropout(0.25)(cat_in)

cat_out = LSTM(300,return_sequences=False , kernel_regularizer= l1_l2(0.01))(cat_drop)

output_attention_mul = Lambda(lambda x : tf.multiply(x,cat_out))(desc_in)

o_mul_drop = Dropout(0.3)(output_attention_mul)

lstm_out =Bidirectional( LSTM(100 , return_sequences=False , kernel_regularizer= l1_l2(0.01)))(o_mul_drop)

output = Dense(1, activation='sigmoid')(lstm_out)

model = Model(inputs = [desc_input , cat_inp] , outputs = [output])

model.compile(optimizer='rmsprop', loss='binary_crossentropy' , metrics=['acc'])



train_categories = pickle.load(open('./pkl/categories.p','rb'))
train_descriptions = pickle.load(open('./pkl/descriptions.p','rb'))
train_labels = pickle.load(open('./pkl/labels.p','rb'))

l = np.arange(0 , len(train_descriptions))

train = np.copy(l)
count = 0

for z in range(0 ,60):
    print("Here:",count)
    np.random.shuffle(train)
    for i in train:
        count+=1
        model.fit([np.array(train_descriptions[i])[np.newaxis, :] , np.array(train_categories[i])[np.newaxis, :]], [np.array([train_labels[i]])[:,np.newaxis]],
              epochs=1, batch_size=1 , shuffle=True )

print("\n\n\n\n\n\n")

output= []
for i in l[160:179]:
    count += 1
    o = model.predict([np.array(train_descriptions[i])[np.newaxis, :], np.array(train_categories[i+1])[np.newaxis, :]])
    print(o, i)
    output.append(o)

pickle.dump(l, open('./pkl/target.p' ,  'wb'))
pickle.dump(o , open('./pkl/predicted.p' , 'wb'))


