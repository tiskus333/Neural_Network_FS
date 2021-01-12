#!/usr/bin/python3
import numpy as np
from random import random
class ReLU:
    def forward(self,inputs):
        return np.maximum(0,inputs)
    def backward(self,inputs):
        return np.where(inputs >= 0, 1, 0)

class Sigmoid:
    def forward(self,inputs):
        return 1.0/(1.0+np.exp(np.negative(inputs)))
    def backward(self,inputs):
        sig = self.forward(inputs)
        return  sig * (1-sig)

def MSE(outputs, targets):
    return (np.square(np.subtract(targets,outputs))).mean(axis=1)

class NeuralNetwork:
    def __init__(self,input_layer,hidden_layers = [3,3],output_layer=1,activation_function="ReLU"):
        layers = [input_layer] + hidden_layers + [output_layer]
        self.weights = [np.random.rand(layers[i],layers[i+1])*0.1 for i in range(len(layers)-1)]
        self.biases = [np.zeros((1,layers[i+1])) for i in range(len(layers)-1)]
        self.activations = [np.zeros((1,layers[i])) for i in range(len(layers))]
        self.derivatives = [np.zeros((1,layers[i])) for i in range(len(layers)-1)]

        if activation_function == "ReLU":
            self.activation_function = ReLU()
        elif activation_function == "SIGMOID":
            self.activation_function = Sigmoid()
        else:   
            print("No such function")
    
    def train(self,train_inputs,targets,batch_size,epochs,learning_rate):
        for i in range(epochs):
            l = []
            j=0
            while j<len(targets):
                input_batch = train_inputs[j:j+batch_size]
                target_batch = targets[j:j+batch_size]
                out = self.forward_propagate(input_batch)
                error = out - target_batch
                l.append(MSE(out,target_batch))
                self.back_propagate(error,learning_rate)
                j += batch_size

            accuracy = np.average(l)
            if (i+1)%10 == 0:
                print("epochs:", i + 1, "==== error:", np.sum(MSE(out,target_batch)))  


    def predict(self,inputs):
        return self.forward_propagate(inputs)
    
    def forward_propagate(self,inputs):
        activation = inputs
        self.activations[0] = inputs
        for i, (weight, bias) in enumerate(zip(self.weights,self.biases)):
            layer_output = np.dot(activation,weight) + bias
            activation = self.activation_function.forward(layer_output)
            self.activations[i+1] = activation
        return activation

    def back_propagate(self,error,learning_rate):
        for i in reversed(range(len(self.derivatives))):
            delta = error * self.activation_function.backward(self.activations[i+1])
            self.derivatives[i] = np.dot(self.activations[i].T,delta)
            error = np.dot(delta,self.weights[i].T)
            self.biases[i] -= np.sum(delta,axis=0,keepdims = True)
            self.weights[i] -= learning_rate*self.derivatives[i]

test1 = np.array([[random()/2 for _ in range(2)] for _ in range(3000)])
targets = np.array([[i[0]*i[1]] for i in test1])

nn = NeuralNetwork(2,[10,4],1,"SIGMOID")

inputs = [[0.3,0.1],[0.3,0.4],[0.5,0.1],[0.2,0.3],[-1,-2]]
results = nn.predict(inputs)
for num in range(len(results)):
   print("Num: "+str(round(results[num][0],5))+" For " + str(inputs[num][0])+"*"+str(inputs[num][1]))

#https://www.kdnuggets.com/2019/08/numpy-neural-networks-computational-graphs.html