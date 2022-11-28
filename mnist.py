import pickle
from sklearn.datasets import fetch_openml
import numpy as np
from pathlib import Path

Path("dataset").mkdir(exist_ok=True)


def download_mnist():
    """
    Download MNIST dataset and save it as mnist.data in the dataset directory.
    """
    X, y = fetch_openml("mnist_784", return_X_y=True)
    X = X.to_numpy()
    X = X / 255
    digits = {}
    for i in range(10):
        digits[str(i)] = []
    for i in range(len(y)):
        digits[y[i]].append(X[i])
    path = "dataset/mnist.data"
    f = open(path, 'wb')
    pickle.dump(digits, f)
    f.close()


def load_mnist():
    """
    Returns P and Q_list where P consists of images of all digits 
    in mnist.data, and Q_list contains 5 elements each consisting
    of images of fewer digits.
    This function should only be run after download_mnist().
    """
    with open('dataset/mnist.data', 'rb') as handle:
        X = pickle.load(handle)
    P  = np.vstack(
        (X['0'], X['1'], X['2'], X['3'], X['4'], X['5'], X['6'], X['7'], X['8'], X['9'])
    )
    Q1 = np.vstack((X['1'], X['3'], X['5'], X['7'], X['9']))
    Q2 = np.vstack((X['0'], X['1'], X['3'], X['5'], X['7'], X['9']))
    Q3 = np.vstack((X['0'], X['1'], X['2'], X['3'], X['5'], X['7'], X['9']))
    Q4 = np.vstack((X['0'], X['1'], X['2'], X['3'], X['4'], X['5'], X['7'], X['9']))
    Q5 = np.vstack(
        (X['0'], X['1'], X['2'], X['3'], X['4'], X['5'], X['6'], X['7'], X['9'])
    )
    Q_list = [Q1, Q2, Q3, Q4, Q5]
    return P, Q_list

