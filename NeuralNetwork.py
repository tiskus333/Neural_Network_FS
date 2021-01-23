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

def softmax(scores):
    scores -= np.max(scores)
    prob = (np.exp(scores).T / np.sum(np.exp(scores), axis=1)).T
    return prob

def cross_entropy_loss(output,label):
    m = label.shape[0]
    loss = -np.sum(label*np.log(output + 1e-12))/m
    return loss

def MSE(outputs, targets):
    return (np.square(np.subtract(targets,outputs))).mean(axis=1)

class NeuralNetwork:
    def __init__(self,input_layer,hidden_layers = [3,3],output_layer=1,activation_function="ReLU"):
        self.layers = [input_layer] + hidden_layers + [output_layer]
        self.weights = [np.random.uniform(-1/np.sqrt(self.layers[i]),1/np.sqrt(self.layers[i+1]),size=(self.layers[i],self.layers[i+1])) for i in range(len(self.layers)-2)]
        self.weights.append(np.zeros((self.layers[-2],self.layers[-1])))
        print("weights:\n",self.weights)
        self.biases = [np.zeros((1,self.layers[i+1])) for i in range(len(self.layers)-1)]
        print("biases:\n",self.biases)
        self.activations = [np.zeros((1,self.layers[i])) for i in range(len(self.layers))]
        print("activations:\n",self.activations)
        self.derivatives = [np.zeros((1,self.layers[i])) for i in range(len(self.layers)-1)]
        print("derivatives:\n",self.derivatives,'\n')

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
                error = cross_entropy_loss(out,target_batch)
                print("error:\n",error)
                l.append(error)
                self.back_propagate(error,learning_rate)
                j += batch_size

            if (i+1)%10 == 0:
                accuracy = np.average(l)
                print("epochs:", i + 1, "==== error:", accuracy)  


    def predict(self,inputs):
        return self.forward_propagate(inputs)
    
    def forward_propagate(self,inputs):
        activation = inputs
        self.activations[0] = inputs
        print("inputs \n",inputs)
        for i in range(len(self.layers)-2):
            layer_output = np.dot(activation,self.weights[i]) + self.biases[i]
            print("layer output", i, '\n', layer_output)
            activation = self.activation_function.forward(layer_output)
            self.activations[i+1] = activation
            print("activation layer ", i, '\n', activation)
        layer_output = np.dot(activation,self.weights[-1]) + self.biases[-1]
        print("Last layer output\n", layer_output)
        activation = softmax(layer_output)
        self.activations[-1] = activation
        print("Softmax activation layer\n", activation)
        return activation

    def back_propagate(self,error,learning_rate):
        for i in reversed(range(len(self.derivatives))):
            delta = error * self.activation_function.backward(self.activations[i+1])
            print("delta:\n",delta)
            self.derivatives[i] = np.dot(delta,self.activations[i].T)
            error = np.dot(delta,self.weights[i].T)
            # self.biases[i] -= np.sum(delta,axis=0,keepdims = True)
            self.weights[i] -= learning_rate*self.derivatives[i]
            print("weights\n",self.weights)

test1 = np.array([[random()/2 for _ in range(2)] for _ in range(3000)])
targets = np.array([[i[0]*i[1]] for i in test1])

nn = NeuralNetwork(4,[3],2,"ReLU")
nn.train(np.array([[0,0,0,0],[0,0,1,0],[1,0,0,0],[1,1,0,0],[1,1,0,1]]),np.array([[1,0],[0,1],[0,1],[1,0],[1,0]]),1,1,0.1)
# inputs = [[0.3,0.1],[0.3,0.4],[0.5,0.1],[0.2,0.3],[-1,-2]]
results = nn.predict([0,1,0,0])
print("RESULTS: ",results)
# for num in range(len(results)):
#    print("Num: "+str(round(results[num][0],5))+" For " + str(inputs[num][0])+"*"+str(inputs[num][1]))

#https://www.kdnuggets.com/2019/08/numpy-neural-networks-computational-graphs.html

# print(cross_entropy_loss(np.array([[0.25,0.25,0.25,0.25],[0.01,0.01,0.01,0.96]]),np.array([[0,0,0,1],[0,0,0,1]])))