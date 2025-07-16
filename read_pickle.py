import pickle

file_path = './data/deepcad_test_mesh/train.pkl'
with open(file_path, 'rb') as file:
    data = pickle.load(file)

print(data)