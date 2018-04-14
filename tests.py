import numpy as np
import pickle
from keras.models import load_model

l = pickle.load( open('./pkl/target.p',  'rb'))
o = pickle.load(open('./pkl/predicted.p' , 'rb'))

print(len(l) , len(o))
print(l)
print(o)
