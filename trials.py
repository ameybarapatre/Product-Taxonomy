import pandas as pd
import numpy as np
import json
import pickle
"""
data= pd.read_csv("./Data/labeled_processed.csv")
zeros = pd.read_csv("./Data/synthetic.csv").sample(90)

data = data.append(zeros, ignore_index=True)

data.to_csv('./Data/balanced.csv')
"""

print(pickle.load(open('./pkl/descriptions.p' ,'rb')))