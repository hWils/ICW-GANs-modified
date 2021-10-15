import pickle
import pandas as pd

fin = 'train_data_list.pkl'
fout = 'test_data_list.pkl'
devfile = 'dev_data_list.pkl'

train = pickle.load(open(fin, 'rb'), encoding='latin1')
test = pickle.load(open(fout, 'rb'), encoding='latin1')
dev = pickle.load(open(devfile, 'rb'), encoding='latin1')
print(len(test), len(train), type(test))
print(dev)