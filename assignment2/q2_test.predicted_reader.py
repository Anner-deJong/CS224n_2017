import pickle

with open ('q2_test.predicted.pkl', 'rb') as fp:
    raw_data = pickle.load(fp)

print(len(raw_data))

print(raw_data)
