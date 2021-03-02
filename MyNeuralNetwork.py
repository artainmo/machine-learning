import numpy as np
from random import randint
import pickle
import matplotlib
matplotlib.use('TkAgg') #Make matplotlib compatible with Big Sur on mac
import matplotlib.pyplot as mpl
from .activation_functions import *
from .init_neural_network import *
from .cost_functions import *
from .manipulate_data import *


def save_neural_network(NN):
    neural_net_params = [NN.name, NN.inputs, NN.expected, NN.test_set_x, NN.test_set_y, NN.deep_layers, NN.alpha, NN.n_cycles, NN._gradient_descend, NN.b, NN._activation_function_layers, NN._activation_function_output, NN._weight_init, NN._cost_function, NN.early_stopping, NN.validation_hold_outset, NN.momentum, NN.feedback, NN.weights, NN.bias]
    pickle.dump(neural_net_params, open("saved/neural_network.pkl", 'wb', closefd=True))
    print("Neural Network saved in saved/neural_network.pkl")

def load_neural_network(path):
    neural_net_params = pickle.load(open(path, 'rb', closefd=True))
    return MyNeuralNetwork(neural_net_params[0], neural_net_params[1], neural_net_params[2], neural_net_params[3], neural_net_params[4], neural_net_params[5], neural_net_params[6], neural_net_params[7], neural_net_params[8], neural_net_params[9], neural_net_params[10], neural_net_params[11], neural_net_params[12], neural_net_params[13], neural_net_params[14], neural_net_params[15], neural_net_params[16], neural_net_params[17], neural_net_params[18], neural_net_params[19])

def show_object(name, obj):
    print(name + ":")
    for elem in obj:
        print(elem.shape)
    print("----------")

def get_mini_batch(inputs, expected, b):
    length = len(inputs)
    last = 0
    pos = 0
    while True:
        pos += b
        if pos > length:
            ret = inputs[last:length]
            pos -= length
            while pos > length:
                pos -= length
                np.concatenate((ret, inputs))
            yield np.concatenate((ret, inputs[0:pos])), np.concatenate((expected[last:length], expected[0:pos]))
        else:
            yield inputs[last:pos], expected[last:pos]
        last = pos

class MyNeuralNetwork():
    def __init__(self, name, inputs, expected, test_set_x=None, test_set_y=None, deep_layers=2, learning_rate=0.01, n_cycles=1000, gradient_descend="mini-batch", b=32, activation_function_layers="tanh", activation_function_output="softmax", weight_init="xavier", cost_function="CE", early_stopping=False, validation_hold_outset="Default", momentum=False, feedback=True, weights=None, bias=None):
        self.name = name
        self._gradient_descend = gradient_descend
        self._activation_function_layers = activation_function_layers
        self._activation_function_output = activation_function_output
        self._weight_init = weight_init
        self._cost_function = cost_function
        if gradient_descend == "stochastic":
            self.gradient_descend = self.__stochastic
        elif gradient_descend == "batch":
            self.gradient_descend = self.__batch
        elif gradient_descend == "mini-batch":
            self.gradient_descend = self.__mini_batch
        else:
            print("Error: My_Neural_Network gradient descend, choose between stochastic, batch, mini-batch")
            exit()
        if activation_function_layers == "sigmoid":
            self.layers_activation_function = sigmoid
            self.derivative_layers_activation_function = derivative_sigmoid
        elif activation_function_layers == "tanh":
            self.layers_activation_function = tanh
            self.derivative_layers_activation_function = derivative_tanh
        elif activation_function_layers == "relu":
            self.layers_activation_function = call_relu
            self.derivative_layers_activation_function = call_derivative_relu
        else:
            print("Error: My_Neural_Network activation function layers, choose between sigmoid, tanh and relu")
            exit()
        if activation_function_output == "sigmoid":
            self.output_activation_function = sigmoid
            self.derivative_output_activation_function = derivative_sigmoid
            self.probabilities_to_answer = sigmoid_to_answer
        elif activation_function_output == "softmax":
            self.output_activation_function = softmax
            self.derivative_output_activation_function = derivative_softmax
            self.probabilities_to_answer = softmax_to_answer
        elif activation_function_output == "relu":
            self.output_activation_function = call_relu
            self.derivative_output_activation_function = call_derivative_relu
            self.probabilities_to_answer = relu_to_answer
        else:
            print("Error: My_Neural_Network activation function output, choose between sigmoid, softmax and relu")
            exit()
        if weight_init == "xavier":
            weight_init = xavier
        elif weight_init == "he":
            weight_init = he
        elif weight_init == None:
            weight_init = normal
        else:
            print("Error: weight init type, choose between xavier, he and None")
            exit()
        if cost_function == "MSE":
            self.cost_function = mean_square_error
            self.derivative_cost_function = derivative_mean_square_error
        elif cost_function == "CE":
            self.cost_function = cross_entropy
            self.derivative_cost_function = derivative_cross_entropy
        else:
            print("Error: cost function, choose between MSE and CE")
            exit()
        self.inputs = inputs
        self.expected = expected
        self.deep_layers = deep_layers
        self.layers = init_layers(self.deep_layers + 1, inputs.shape[1], self.expected.shape[1])
        if weights == None:
            self.weights = init_weights(self.layers, inputs.shape[1], self.expected.shape[1], weight_init)
        else:
            self.weights = weights
        if bias == None:
            self.bias = init_bias(self.weights)
        else:
            self.bias = bias
        self.__reset_gradients()
        self.alpha = learning_rate
        self.n_cycles = n_cycles
        self.b = b #mini-batch size
        self.costs = []
        self.costs_test_set = []
        self.feedback = feedback
        self.test_set_x = test_set_x
        self.test_set_y = test_set_y
        if momentum == True:
            self.gamma = 0.9
        else:
            self.gamma = 0
        self.momentum = momentum
        self.velocity_weights = copy_object_shape(self.weights)
        self.velocity_bias = copy_object_shape(self.bias)
        if early_stopping == True and self.test_set_x is not None and self.test_set_y is not None:
            self.early_stopping = True
            if validation_hold_outset == "Default":
                self.validation_hold_outset = int(self.inputs.shape[0] / 100 * 10)
            else:
                self.validation_hold_outset = validation_hold_outset
            self.cost_rising = 0
            self.lowest_cost_index = 0
            self.best_weights = copy_object_shape(self.weights)
            self.best_bias = copy_object_shape(self.bias)
        else:
            self.early_stopping = False
            self.validation_hold_outset = "Default"
        if self.feedback == True:
            self.show_all()

    def show_all(self):
        print("--------------------------------------DEEP NEURAL NETWORK STRUCTURE--------------------------------------")
        print("NEURAL NETWORK NAME -> " + str(self.name))
        show_object("Layer", self.layers)
        show_object("Weight", self.weights)
        show_object("Bias", self.bias)
        show_object("Output gradient weight", self.output_gradient_weight)
        show_object("Output gradient bias", self.output_gradient_bias)
        show_object("Deep gradient weight", self.deep_gradient_weight)
        show_object("Deep gradient bias", self.deep_gradient_bias)
        print("---------------------------------------------------------------------------------------------------------")

    #If no lowering of costs compared to lowest cost after 50epochs, early stop and whenever stopping always keep weights and bias associated with lowest cost and cut graphs until lowest cost
    def __early_stopping(self, epoch):
        if epoch == 1:
            self.lowest_cost_index = epoch - 1
            self.cost_rising = 0
        elif self.costs_test_set[self.lowest_cost_index] > self.costs_test_set[-1]:
            self.lowest_cost_index = epoch - 1
            self.best_weights = self.weights
            self.best_bias = self.bias
            self.cost_rising = 0
        else:
            self.cost_rising += 1
        if self.cost_rising >= self.validation_hold_outset or (epoch == self.n_cycles and self.lowest_cost_index != epoch - 1):
            self.weights = self.best_weights
            self.bias = self.best_bias
            return 1
        return 0

    def cost(self, epoch=None, feedback=False): #cost function calculates total error of made prediction, mean over output nodes
        total_error = np.sum([self.cost_function(predicted, expected) for predicted, expected in zip(self.predict(self.inputs, probabilities_to_answer=False), self.expected)]) / self.inputs.shape[0]
        self.costs.append(total_error)
        if self.test_set_x is not None and self.test_set_y is not None:
            total_error_test = np.sum([self.cost_function(predicted, expected) for predicted, expected in zip(self.predict(self.test_set_x, probabilities_to_answer=False), self.test_set_y)]) / self.test_set_x.shape[0]
            self.costs_test_set.append(total_error_test)
            if feedback == True:
                print("Epoch: " + str(epoch) + "/" + str(self.n_cycles) + " -> Cost: " + str(total_error) + " --> Test set Cost: " + str(total_error_test))
        elif feedback == True:
            print("Epoch: " + str(epoch) + "/" + str(self.n_cycles) + " -> Cost: " + str(total_error))
        return total_error

    def basic_graph(self):
        mpl.plot(range(len(self.costs)), self.costs, label=str(self.name) + " training set")
        if self.early_stopping == True:
            mpl.plot(range(len(self.costs[0:self.lowest_cost_index])), self.costs[0:self.lowest_cost_index], label=str(self.name) + " training set stop")
        if self.test_set_x is not None and self.test_set_y is not None:
            mpl.plot(range(len(self.costs_test_set)), self.costs_test_set, label=str(self.name) + " test set")
            if self.early_stopping == True:
                mpl.plot(range(len(self.costs_test_set[0:self.lowest_cost_index])), self.costs_test_set[0:self.lowest_cost_index], label=str(self.name) + " test set stop")

    def __feedback_cost_graph(self):
        input("========================\nPress Enter To See Graph\n========================")
        self.basic_graph()
        mpl.title("Starting Cost: " + str(round(self.costs[0], 5))  + "\nFinal Cost: " + str(round(self.costs[-1], 5)))
        mpl.legend()
        mpl.show()

    def forward_propagation(self, inputs):
        self.layers[0] = np.array([inputs], dtype=np.float128)
        for i in range(len(self.layers) - 2):
            self.layers[i + 1] = self.layers_activation_function(np.dot(self.layers[i], self.weights[i]) + self.bias[i])
        self.layers[-1] = self.output_activation_function((np.dot(self.layers[-2], self.weights[-1]) + self.bias[-1]))
        self.predicted = self.layers[-1]

    def __output_layer_partial_derivatives(self, expected):
        Delta = self.derivative_cost_function(self.predicted, expected) * self.derivative_output_activation_function(self.predicted)
        return np.dot(self.layers[-2].T, Delta), Delta

    def __deep_layer_partial_derivatives(self, position, Delta):
        Delta = (np.dot(self.weights[position + 1], Delta.T) * (self.derivative_layers_activation_function(self.layers[position + 1])).T).T
        return np.dot(self.layers[position].T, Delta), Delta

    def backward_propagation(self, expected):
        gradient, Delta = self.__output_layer_partial_derivatives(expected)
        self.output_gradient_weight[0] = self.output_gradient_weight[0] + gradient
        self.output_gradient_bias[0] = self.output_gradient_bias[0] + Delta #bias weight does not need to get multiplied by prior bias node as it is equal to one
        for i in range(len(self.weights) - 2, -1, -1): #range starts from last non-output weights until first weights (index 1)
            gradient, Delta = self.__deep_layer_partial_derivatives(i, Delta)
            self.deep_gradient_weight[i] = self.deep_gradient_weight[i] + gradient
            self.deep_gradient_bias[i] = self.deep_gradient_bias[i] + Delta

    def __reset_gradients(self):
        self.output_gradient_weight = copy_object_shape([self.weights[-1]])
        self.output_gradient_bias = copy_object_shape([self.bias[-1]])
        self.deep_gradient_weight = copy_object_shape(self.weights[0:-1])
        self.deep_gradient_bias = copy_object_shape(self.bias[0:-1])

    def __momentum(self):
        momentum_weights = copy_object_shape(self.weights)
        momentum_bias = copy_object_shape(self.bias)
        if self.momentum == False:
            return momentum_weights, momentum_bias
        for i in range(len(self.weights)):
            if i == len(self.weights) - 1:
                self.velocity_weights[i] = self.velocity_weights[i] + (self.alpha * self.output_gradient_weight[0])
                momentum_weights[i] = self.gamma * self.velocity_weights[i]
                self.velocity_bias[i] = self.velocity_bias[i] + (self.alpha * self.output_gradient_bias[0])
                momentum_bias[i] = self.gamma * self.velocity_bias[i]
            else:
                self.velocity_weights[i] = self.gamma * self.velocity_weights[i] + (self.alpha * self.deep_gradient_weight[i])
                momentum_weights[i] = self.gamma * self.velocity_weights[i]
                self.velocity_bias[i] = self.gamma * self.velocity_bias[i] + (self.alpha * self.deep_gradient_bias[i])
                momentum_bias[i] = self.gamma * self.velocity_bias[i]
        return momentum_weights, momentum_bias

    def __update_weights(self, _epoch):
        momentum_weights, momentum_bias = self.__momentum()
        self.weights[-1] = self.weights[-1] - ((self.alpha * self.output_gradient_weight[0]) + momentum_weights[-1])
        self.bias[-1] = self.bias[-1] - ((self.alpha * self.output_gradient_bias[0]) + momentum_bias[-1])
        for i in range(len(self.weights) - 2, -1, -1): #range starts from last non-output weights until first weights (index 0)
            self.weights[i] = self.weights[i] - ((self.alpha * self.deep_gradient_weight[i]) + momentum_weights[i])
            self.bias[i] = self.bias[i] - ((self.alpha * self.deep_gradient_bias[i]) + momentum_bias[i])
        self.__reset_gradients()
        self.cost(epoch=_epoch, feedback=self.feedback)

    def __cycle(self, inputs, expected):
         self.forward_propagation(inputs)
         self.backward_propagation(expected)

    def __batch(self):
        for i in range(self.n_cycles):
            for inputs, expected in zip(self.inputs, self.expected):#complete batch cycle
                self.__cycle(inputs, expected)
            self.__update_weights(i + 1)
            if self.early_stopping == True and self.__early_stopping(i + 1):
                print("Early Stopping was used")
                break

    def __mini_batch(self):
        generator = get_mini_batch(self.inputs, self.expected, self.b)
        for i in range(self.n_cycles):
            inputs, expected = next(generator)
            for _inputs, _expected in zip(self.inputs, self.expected):#complete batch cycle
                self.__cycle(_inputs, _expected)
            self.__update_weights(i + 1)
            if self.early_stopping == True and self.__early_stopping(i + 1):
                print("Early Stopping was used")
                break

    def __stochastic(self):
        length = len(self.inputs) - 1
        for i in range(self.n_cycles):
            random = randint(0, length)
            self.__cycle(self.inputs[random], self.expected[random])
            self.__update_weights(i + 1)
            if self.early_stopping == True and self.__early_stopping(i + 1):
                print("Early Stopping was used")
                break


    def fit(self):
        input("=============================\nPress Enter To Start Training\n=============================")
        self.costs.clear()
        self.costs_test_set.clear()
        self.__reset_gradients()
        self.gradient_descend()
        if self.feedback == True:
            self.__feedback_cost_graph()

    def predict(self, inputs, probabilities_to_answer=True):
        answers = np.zeros((inputs.shape[0], self.expected.shape[1]))
        for i in range(inputs.shape[0]):
            self.forward_propagation(inputs[i])
            answers[i] = self.predicted
        if probabilities_to_answer == True:
            return self.probabilities_to_answer(answers)
        else:
            return answers

    def training_metric_history(self):
        input("================================================\nPress Enter To View Training Cost Metric History\n================================================")
        if self.test_set_x is not None and self.test_set_y is not None:
            for epoch in range(len(self.costs)):
                print("Epoch: " + str(epoch + 1) + "/" + str(self.n_cycles) + " -> Cost: " + str(self.costs[epoch]) + " --> Test set Cost: " + str(self.costs_test_set[epoch]))
        else:
            for epoch in range(len(self.costs)):
                print("Epoch: " + str(epoch + 1) + "/" + str(self.n_cycles) + " -> Cost: " + str(self.costs[epoch]))


    def __gradients_to_vector(self):
        back_prop_gradient = np.concatenate((self.output_gradient_weight[0].flatten(), self.output_gradient_bias[0].flatten()))
        for g_w, g_b in zip(self.deep_gradient_weight, self.deep_gradient_bias):
            back_prop_gradient = np.concatenate((back_prop_gradient, g_w.flatten(), g_b.flatten()))
        return back_prop_gradient

    def __vectorize_numerical_gradients(self, inputs, expected, epsilon=1e-4):
        numerical_gradient = np.array([])
        for i in range(len(self.weights)):
            for l in range(self.weights[i].shape[0]):
                for k in range(self.weights[i].shape[1]):
                    rem = self.predicted
                    self.weights[i][l][k] = self.weights[i][l][k] + epsilon
                    self.forward_propagation(inputs)
                    Jmax = self.cost_function(self.predicted, expected)
                    self.weights[i][l][k] = self.weights[i][l][k] - (2*epsilon)
                    self.forward_propagation(inputs)
                    Jmin = self.cost_function(self.predicted, expected)
                    self.weights[i][l][k] = self.weights[i][l][k] + epsilon
                    self.predicted = rem
                    numerical_gradient = np.append(numerical_gradient, (Jmax - Jmin) / (2*epsilon))
            for l in range(self.bias[i].shape[1]):
                rem = self.predicted
                self.bias[i][0][l] = self.bias[i][0][l] + epsilon
                self.forward_propagation(inputs)
                Jmax = self.cost_function(self.predicted, expected)
                self.bias[i][0][l] = self.bias[i][0][l] - (2*epsilon)
                self.forward_propagation(inputs)
                Jmin = self.cost_function(self.predicted, expected)
                self.bias[i][0][l] = self.bias[i][0][l] + epsilon
                self.predicted = rem
                numerical_gradient = np.append(numerical_gradient, (Jmax - Jmin) / (2*epsilon))
        return numerical_gradient

    def check_gradients(self):
        for i in range(10):
            random = randint(0, self.inputs.shape[0] - 1)
            self.__reset_gradients()
            self.__cycle(self.inputs[random], self.expected[random]) #Compute backprop gradients
            back_prop_gradient = np.sqrt(np.square(self.__gradients_to_vector())) #gradients to vector, make them all positive
            numerical_gradient = np.sqrt(np.square(self.__vectorize_numerical_gradients(self.inputs[random], self.expected[random]))) #numerical gradients to vector make them all positive
            difference = np.copy(back_prop_gradient)
            for i in range(back_prop_gradient.shape[0]):
                if back_prop_gradient[i] > numerical_gradient[i]:
                    difference[i] = back_prop_gradient[i] - numerical_gradient[i]
                else:
                    difference[i] = numerical_gradient[i] - back_prop_gradient[i]
            relative_error = minmax_normalization(difference) / (minmax_normalization(back_prop_gradient) + minmax_normalization(numerical_gradient))
            for rel_err in relative_error:
                if rel_err > 1e-4:
                    print("Gradient check wrong" + " - difference: " + str(rel_err))
                else:
                    print("Gradient check correct"+ " - difference: " + str(rel_err))
        input("=======================\nPress Enter To Continue\n=======================")


# if __name__ == "__main__":
#     x = np.array([[0,0,1,1,0,0],[0,1,1,1,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0]]) #4X3 -> 4 examples and 3 inputs expected
#     y = np.array([[0, 1],[1, 1],[1, 0],[1, 0]]) #4X2 -> 4 examples and 2 outputs expected
#     test = MyNeuralNetwork(x, y)
# test.fit()
