import pickle


def read_pickle(file_name="../datasets/MT10.pkl"):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    read_pickle()
