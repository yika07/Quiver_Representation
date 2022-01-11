import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable


class GetData:
    def __init__(self):
        pass

    @staticmethod
    def iris_dataset():
        iris = load_iris()
        x = iris['data']
        y = iris['target']
        x_scaled = StandardScaler().fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=2)
        x_train = Variable(torch.from_numpy(x_train)).float()
        y_train = Variable(torch.from_numpy(y_train)).long()
        x_test = Variable(torch.from_numpy(x_test)).float()
        y_test = Variable(torch.from_numpy(y_test)).long()

        return x_train, y_train, x_test, y_test
