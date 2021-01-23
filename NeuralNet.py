import numpy as np

class DenseLayer:
    def __init__(self, n_inputs, n_neurons,activation_function = "relu"):
        self.weights = np.random.uniform(-1., 1., size=(n_inputs,n_neurons)).astype(np.float16)/np.sqrt(n_inputs*n_neurons)
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
        return np.maximum(0,inputs)
    def backward(self,inputs):
        return np.where(inputs >= 0, 1, 0)

class Sigmoid:
    def forward(self,inputs):
        return 1.0/(1.0+np.exp(np.negative(inputs)))
    def backward(self,inputs):
        sig = self.forward(inputs)
        return  sig * (1-sig)

class Softmax:
    def forward(self,inputs):
        """Compute the softmax of vector x."""
        # exps = np.exp(inputs - inputs.max())
        # return exps / np.sum(exps)
        inputs -= np.max(inputs)
        prob = (np.exp(inputs).T / np.sum(np.exp(inputs), axis=1)).T
        return prob
    def backward(self,inputs):
        return  inputs*(inputs -(inputs * inputs).sum(axis=1)[:,None])

def MSE(outputs, targets):
    return (np.square(np.subtract(targets,outputs))).mean(axis=1)

def cross_entropy_loss(output,label):
    m = label.shape[0]
    loss = -np.sum(label*np.log(output + 1e-12))/m
    return np.squeeze(loss)


class NeuralNetwork:
    def __init__(self,input_layer,hidden_layers = [3,3],output_layer=1,activation_function="relu"):
        self.layers = []
        prev_layer_size = input_layer
        for layer_size in hidden_layers:
            self.layers.append(DenseLayer(prev_layer_size,layer_size,activation_function))
            prev_layer_size = layer_size
        self.layers.append(DenseLayer(prev_layer_size,output_layer,"softmax"))
        for l in self.layers:
            print(l.weights)
    
    def train(self,train_data,targets,epochs=1000,batch_size=1,learning_rate=0.01):
        for i in range(epochs):
            l = []
            for j in range(0,len(train_data),batch_size):
                input_batch = train_data[j:j+batch_size]
                target_batch = targets[j:j+batch_size]
                out = self.forward_propagate(input_batch)
                error = cross_entropy_loss(out,target_batch)
                loss = error.mean()
                l.append(loss)
                self.back_propagate(input_batch,error)
                self.gradient_descent(learning_rate)
            if (i+1)%10 == 0:
                print("epochs:", i + 1, "==== error:", loss)  


    def test(self,test_data):
        pass

    def predict(self,data):
        return self.forward_propagate(data)
    
    def forward_propagate(self,data):
        tmp_data = data
        for layer in self.layers:
            tmp_data = layer.forward(tmp_data)
        return tmp_data

    def back_propagate(self,inputs,error):
        layer2_error = error
        layer2_delta = layer2_error * self.layers[1].activation_function.backward(self.layers[1].activations)
        self.layers[1].derivatives = np.dot(self.layers[0].activations.T,layer2_delta)


        layer1_error = np.dot(layer2_delta,self.layers[1].weights.T)
        layer1_delta = layer1_error * self.layers[0].activation_function.backward(self.layers[0].activations)
        self.layers[0].derivatives = np.dot(inputs.T,layer1_delta)

        # for layer in reversed(self.layers):
        #     activation = layer.activations
        #     delta = error * layer.activation_function.backward(activation)
        #     layer.derivatives = np.dot(activation,delta)
        #     error = np.dot(delta,layer.weights.T)
        # return error

    def gradient_descent(self,learning_rate):
        for layer in self.layers:
            #layer.biases -= layer.derivatives
            layer.weights -= layer.derivatives*learning_rate

# test1 = np.random.rand(3000,2)
# test1 /= 2
# targets = np.array([[i[0]+i[1]] for i in test1])
test2 = np.random.randint(0,2,(1000,4))
target2 = np.zeros((1000,2))
for x,y in zip(test2,target2):
    if x[0] + x[2] == 2 and x[1] == 0:
        y[1] = 1
    else:
        y[0] = 1


nn = NeuralNetwork(4,[3],2,"sigmoid")
nn.train(test2,target2,100,10,0.1)
print("RESULTS: ")
print(nn.predict([0,1,1,0]))
print(nn.predict([0,0,1,0]))
print(nn.predict([0,1,1,1]))
print(nn.predict([1,1,1,1]))
print(nn.predict([0,0,0,0]))


# nn.train(test1,targets,100,0.1)
# print(nn.predict([[0.3,0.1],[0.3,0.4],[0.3,0.1],[0.2,0.3]]))

# for layer in nn.layers:
#     print(layer.weights)

#https://www.kdnuggets.com/2019/08/numpy-neural-networks-computational-graphs.html