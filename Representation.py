import numpy
import numpy as np
import torch


class QuiverRepresentations():

    def __init__(self, data_sample, feature_dimension, nn_parameters, tot_layers, wanted_layer, activation_functions):
        self.data_sample = data_sample
        self.feature_dimension = feature_dimension
        self.nn_parameters = nn_parameters
        self.tot_layers = tot_layers
        self.wanted_layer = wanted_layer
        self.activation_functions = activation_functions

    @staticmethod
    def tanh(vector):
        vector_tensor = torch.from_numpy(vector)
        activated_tensor = torch.tanh(vector_tensor)
        return activated_tensor.numpy()

    @staticmethod
    def sigmoid(vector):
        vector_tensor = torch.from_numpy(vector)
        activated_tensor = torch.sigmoid(vector_tensor)
        return activated_tensor.numpy()

    @staticmethod
    def softmax(vector):
        vector_tensor = torch.from_numpy(vector)
        activated_tensor = torch.softmax(vector_tensor)
        return activated_tensor.numpy()

    @staticmethod
    def relu(vector):
        vector_tensor = torch.from_numpy(vector)
        activated_tensor = torch.relu(vector_tensor)
        return activated_tensor.numpy()

    @staticmethod
    def matrix_embedding(matrix, vector):
        matrix_array = numpy.array(matrix)
        num_rows_matrix = len(matrix_array)
        num_cols_matrix = len(matrix_array[0])
        for j in range(num_cols_matrix):
            a = vector[j]
            for k in range(num_rows_matrix):
                matrix_array[k][j] *= a

        return matrix_array

    @staticmethod
    def output_from_matrix(matrix):
        vector = []
        num_rows_matrix = len(matrix)
        num_cols_matrix = len(matrix[0])
        for i in range(num_rows_matrix):
            a = 0
            for j in range(num_cols_matrix):
                a += matrix[i][j]
            vector.append(a)
        return np.array(vector)

    def embedding_factor(self, vector, function):
        coefficient = []
        if function == 'tanh':
            activated_vector = self.tanh(vector)
            for i in range(len(vector)):
                coefficient.append(activated_vector[i] / vector[i])
            coefficient = np.array(coefficient)
            return coefficient
        if function == 'sigmoid':
            activated_vector = self.sigmoid(vector)
            for i in range(len(vector)):
                coefficient.append(activated_vector[i] / vector[i])
            coefficient = np.array(coefficient)
            return coefficient
        if function == 'relu':
            activated_vector = self.relu(vector)
            for i in range(len(vector)):
                coefficient.append(activated_vector[i] / vector[i])
            coefficient = np.array(coefficient)
            return coefficient
        if function == 'softmax':
            activated_vector = self.softmax(vector)
            for i in range(len(vector)):
                coefficient.append(activated_vector[i] / vector[i])
            coefficient = np.array(coefficient)
            return coefficient
        if function == 'none':
            return vector

    def quiver_space_matrices(self):
        sample = self.data_sample
        parameters = self.nn_parameters
        total_layers = self.tot_layers
        quiver_matrices = []
        function = self.activation_functions
        feature_size = self.feature_dimension
        layer = self.wanted_layer
        for i in range(total_layers):
            a = parameters[i]
            if i == 0:
                weights = self.matrix_embedding(a, sample)
                vec_output = self.output_from_matrix(weights)
                factor_weights = self.embedding_factor(vec_output, function[i])
                quiver_matrices.append(weights)

            else:
                weights = self.matrix_embedding(a, factor_weights)
                vec_output = self.output_from_matrix(weights)
                factor_weights = self.embedding_factor(vec_output, function[i])
                quiver_matrices.append(weights)

        propagation_vector = np.ones(feature_size)
        neuron_values = []
        for i in range(layer):
            activation_function = function[i]
            if i == 0:
                neuron_values = np.dot(quiver_matrices[i], propagation_vector)
            else:
                neuron_values = np.dot(quiver_matrices[i], neuron_values)
        neuron_values = np.array(neuron_values)
        if activation_function == 'tanh':
            activated_neuron_values = self.tanh(neuron_values)
            return neuron_values, activated_neuron_values, quiver_matrices
        if activation_function == 'sigmoid':
            activated_neuron_values = self.sigmoid(neuron_values)
            return neuron_values, activated_neuron_values, quiver_matrices
        if activation_function == 'relu':
            activated_neuron_values = self.relu(neuron_values)
            return neuron_values, activated_neuron_values, quiver_matrices
        if activation_function == 'softmax':
            activated_neuron_values = self.softmax(neuron_values)
            return neuron_values, activated_neuron_values, quiver_matrices
        if activation_function == 'none':
            return neuron_values, neuron_values, quiver_matrices
