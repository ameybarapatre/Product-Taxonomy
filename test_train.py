import pandas as  pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('full_clothing.csv')

print(len(data))

test , train  =  train_test_split(data, test_size=0.1)

print(len(test))
test.to_csv('test.csv')
train.to_csv('train.csv')