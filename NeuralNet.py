#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

class DenseLayer:
    def __init__(self, n_inputs, n_neurons,activation_function = "relu"):
        self.weights = np.random.uniform(-1., 1., size=(n_inputs,n_neurons))/np.sqrt(n_inputs*n_neurons)
        self.biases = np.zeros((1,n_neurons))
        self.derivatives = np.zeros((1,n_neurons))
        self.activations = np.zeros((1,n_neurons))

        if activation_function == "relu":
            self.activation_function = ReLU()
        elif activation_function == "sigmoid":
            self.activation_function = Sigmoid()
        elif activation_function == "softmax":
            self.activation_function = Softmax()
        else:   
            print("No such function")

    def forward(self, inputs):
        layer_output = np.dot(inputs,self.weights) + self.biases
        self.activations = self.activation_function.forward(layer_output)
        return self.activations


class ReLU:
    def forward(self,inputs):
        return np.clip(inputs,0,5000)
        # return np.maximum(0,inputs)
    def backward(self,inputs):
        return np.where(inputs >= 0.0, 1.0, 0.0)

class Sigmoid:
    def forward(self,inputs):
        # return 1.0/(1.0+np.exp(np.negative(inputs)))
        return np.exp(np.fmin(inputs, 0)) / (1 + np.exp(-np.abs(inputs)))
    def backward(self,inputs):
        sig = self.forward(inputs)
        return  sig * (1-sig)

class Softmax:
    def forward(self,inputs):
        # print("softmax: ",inputs)
        z = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        prob = z / np.sum(z, axis=-1,keepdims=True)
        return prob

def cross_entropy_loss(output,label):
    epsilon = 1e-9
    output = np.clip(output,epsilon,1-epsilon)
    loss = -np.sum(label*np.log(output))/output.shape[0]
    return loss

class NeuralNetwork:
    def __init__(self,input_layer,hidden_layers = [3,3],output_layer=1,activation_function="relu"):
        self.error_values = []
        self.layers = []
        prev_layer_size = input_layer
        for layer_size in hidden_layers:
            self.layers.append(DenseLayer(prev_layer_size,layer_size,activation_function))
            prev_layer_size = layer_size
        self.layers.append(DenseLayer(prev_layer_size,output_layer,"softmax"))
        # for l in self.layers:
        #     print(l.weights)
    
    def train(self, train_data, targets, epochs=1000, batch_size=10, learning_rate=0.01, number_of_prints=100):
        for i in range(epochs):
            l = []
            for j in range(0, len(train_data)-len(train_data) % batch_size, batch_size):
                input_batch = train_data[j:j+batch_size]
                target_batch = targets[j:j+batch_size]
                out = self.forward_propagate(input_batch)
                # out = np.nan_to_num(out)
                # print("Output: ",out)
                error = cross_entropy_loss(out, target_batch)
                # print("error: ",error)
                delta = (out - target_batch)/batch_size
                # delta = np.clip(delta,-1,1)
                # delta = np.nan_to_num(delta)
                # print("delta: ",delta)
                l.append(error)
                self.back_propagate(input_batch, delta, learning_rate)
            if number_of_prints != 0 and (i+1) % (int(epochs/number_of_prints)) == 0:
                print("epochs:", i + 1, "==== error:", np.average(l))
            self.error_values.append(np.average(l))

    def test(self,test_data, test_targets):
        sum_t = 0
        pred = self.forward_propagate(test_data)
        ind = pred.argmax(axis=1)
        for i,t in zip(ind,test_targets):
            if t[i] == 1:
                sum_t += 1
        return sum_t/ind.shape[0]

    def predict(self,data, code_back_to_classes = False):
        result = self.forward_propagate(data)
        if code_back_to_classes:
            return result.argmax(axis=1)
        else:
            return result
    
    def forward_propagate(self,data):
        tmp_data = data
        for layer in self.layers:
            tmp_data = layer.forward(tmp_data)
        return tmp_data

    def back_propagate(self,inputs,delta,learning_rate):
        self.layers[-1].derivatives = np.dot(self.layers[-2].activations.T,delta)
        self.layers[-1].biases -= learning_rate * np.sum(delta,axis=0)
        self.layers[-1].weights -= learning_rate * self.layers[-1].derivatives
        for i in reversed(range(len(self.layers) - 1)):
            error = np.dot(delta,self.layers[i+1].weights.T)
            delta = error * self.layers[i].activation_function.backward(self.layers[i].activations)
            if(i == 0):
                self.layers[i].derivatives = np.dot(inputs.T,delta)
            else:
                self.layers[i].derivatives = np.dot(self.layers[i-1].activations.T,delta)
            self.layers[i].biases -= learning_rate * np.sum(delta,axis=0)
            self.layers[i].weights -= learning_rate * self.layers[i].derivatives


if __name__ == "__main__":
    blue = np.random.randn(700, 2) + np.array([0, -3])
    pink = np.random.randn(700, 2) + np.array([3, 3])
    yellow = np.random.randn(700, 2) + np.array([-3, 3])
    feature_set = np.vstack([blue, pink, yellow])
    labels = np.array([0]*700 + [1]*700 + [2]*700)
    one_hot_labels = np.zeros((2100, 3))
    for i in range(2100):
        one_hot_labels[i, labels[i]] = 1

    plt.figure(figsize=(10,7))
    plt.scatter(feature_set[:,0], feature_set[:,1], c=labels, cmap='plasma', s=100, alpha=0.5)

    nn = NeuralNetwork(2,[4],3,"relu")
    nn.train(feature_set,one_hot_labels,epochs=100,batch_size=100,learning_rate=0.001)
    print("RESULTS: ")
    print(nn.predict([0,-3]))
    print(nn.predict([2,2]))
    print(nn.predict([-2,3]))
    print("Accuracy = ", nn.test(feature_set[10:20:1],one_hot_labels[10:20:1]))
    plt.show()

#https://www.kdnuggets.com/2019/08/numpy-neural-networks-computational-graphs.html
