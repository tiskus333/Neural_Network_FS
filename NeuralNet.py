import numpy as np

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.rand(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
        self.activations = np.zeros((1,n_neurons))
        self.derivatives = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

class ReLU:
    def forward(self,inputs):
        return np.maximum(0,inputs)
    def backward(self,inputs):
        return np.where(inputs >= 0, 1, 0)

class Sigmoid:
    def forward(self,inputs):
        return 1/(1+np.exp(inputs))
    def backward(self,inputs):
        sig = self.forward(inputs)
        return  sig * (1-sig)

def MSE(outputs, targets):
    return (np.square(np.subtract(targets,outputs))).mean(axis=1)

class NeuralNetwork:
    def __init__(self,input_layer,hidden_layers = [3,3],output_layer=1,batch_size=1,activation_function="ReLU"):
        self.layers = []
        prev_layer_size = input_layer
        for layer_size in hidden_layers:
            self.layers.append(DenseLayer(prev_layer_size,layer_size))
            prev_layer_size = layer_size
        self.layers.append(DenseLayer(prev_layer_size,output_layer))

        if activation_function == "ReLU":
            self.activation_function = ReLU()
        elif activation_function == "SIGMOID":
            self.activation_function = Sigmoid()
        else:   
            print("No such function")
    
    def train(self,train_data,targets,epochs,learning_rate):
        for i in range(epochs):
            l = []
            for data,target in zip(train_data,targets):
                out = self.forward_propagate(data)
                error = out -target
                l.append(MSE(out,target))
                self.back_propagate(error)
                self.gradient_descent(learning_rate)
            accuracy = (1-(sum(l)/len(train_data)))*100
            if (i+1)%10 == 0:
                print("epochs:", i + 1, "==== accuracy:", accuracy)  


    def test(self,test_data):
        pass

    def predict(self,data):
        return self.forward_propagate(data)
    
    def forward_propagate(self,data):
        tmp_data = self.activation_function.forward(data) 
        for layer in self.layers:
            layer.forward(tmp_data)
            tmp_data = self.activation_function.forward(layer.output)
            layer.activations = tmp_data
        return tmp_data

    def back_propagate(self,error):
        for i in reversed(range(len(self.layers)-1)):
            activation = self.layers[i+1].activations
            delta = error * self.activation_function.backward(activation)
            curr_activation = self.layers[i].activations
            self.layers[i].derivatives = np.dot(curr_activation,delta[0][0])
            error = np.dot(delta[0][0],self.layers[i].weights.T)
        return error

    def gradient_descent(self,learning_rate):
        for layer in self.layers:
            #layer.biases -= layer.derivatives
            layer.weights -= layer.derivatives*learning_rate

test1 = np.random.rand(30000,3)
targets = np.array([[i[0]+i[1]+i[2]] for i in test1])

nn = NeuralNetwork(3,[3,3],1,1,"ReLU")
nn.train(test1,targets,100,0.1)
print(nn.predict([[0.9,0.4,0.1],[0.3,0.4,1],[0.6,0.1,1],[0.2,0.3,1]]))

# for layer in nn.layers:
#     print(layer.weights)

#https://www.kdnuggets.com/2019/08/numpy-neural-networks-computational-graphs.html