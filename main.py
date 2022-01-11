

import sys

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
sys.path.append('../')
from Representation import QuiverRepresentation as qp
from Model import Network
from Data import GetData


def main():
    pass


if __name__ == '__main__':
    main()


x_train, y_train, x_test, y_test = GetData.iris_dataset()
model = Network(x_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)


n_epochs = 100
loss_list = np.zeros((n_epochs, ))
accuracy_list = np.zeros((n_epochs, ))
for epoch in range(n_epochs):
    y_prediction = model(x_train)
    loss = criterion(y_prediction, y_train)
    loss_list[epoch] = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        y_prediction = model(x_test)
        correct = (torch.argmax(y_prediction, dim=1) == y_test).type(torch.FloatTensor)
        accuracy_list[epoch] = correct.mean()


fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex='row')
ax1.plot(accuracy_list)
ax1.set_ylabel("Validation accuracy")
ax2.plot(loss_list)
ax2.set_ylabel("Validation loss")
ax2.set_xlabel("epochs")
plt.show()


parameters_list = []
for name, param in model.named_parameters():
    parameters_list.append(param.data)
for i in range(len(parameters_list)):
    parameters_list[i] = parameters_list[i].numpy()
    parameters_list[i] = tuple(map(tuple, parameters_list[i]))

data_sample = torch.tensor([6.1, 5.7, 3.5, 5.5]).float()
data_x = np.array([6.1, 5.7, 3.5, 5.5])
feat_dimension = 4
nn_parameter = tuple(parameters_list)
total_layers = 3
needed_layer = 3
non_linearity = ['relu', 'relu', 'none']
non_activated_neurons, activated_neurons, matrices = (qp(data_x, feat_dimension, nn_parameter, total_layers,
                                                         needed_layer, non_linearity).quiver_space_matrices())
print(model(data_sample))
print(activated_neurons)

