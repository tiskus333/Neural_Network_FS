import numpy as np
import matplotlib.pyplot as plt

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
        z = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        prob = z / np.sum(z, axis=-1,keepdims=True)
        return prob
    def backward(self,inputs):
        tmp = self.forward(inputs)
        grad_coeff = np.zeros_like(tmp)
        # grad_coeff[np.arange(inputs.shape[0]), y] = -1
        grad_coeff += tmp
        return grad_coeff

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
                l.append(error)
                self.back_propagate(input_batch,out-target_batch,error,learning_rate)
                # self.gradient_descent(learning_rate)
            if (i+1)%10 == 0:
                print("epochs:", i + 1, "==== error:", np.average(l))  


    def test(self,test_data):
        pass

    def predict(self,data):
        return self.forward_propagate(data)
    
    def forward_propagate(self,data):
        tmp_data = data
        for layer in self.layers:
            tmp_data = layer.forward(tmp_data)
        return tmp_data

    def back_propagate(self,inputs,delta,error,learning_rate):
        layer2_error = error
        # layer2_delta = layer2_error * self.layers[1].activations
        layer2_delta = layer2_error * delta
        self.layers[1].derivatives = np.dot(self.layers[0].activations.T,layer2_delta)
        self.layers[1].biases -= learning_rate * np.sum(layer2_delta,axis=0)
        self.layers[1].weights -= learning_rate * self.layers[1].derivatives

        layer1_error = np.dot(layer2_delta,self.layers[1].weights.T)
        layer1_delta = layer1_error * self.layers[0].activation_function.backward(self.layers[0].activations)
        self.layers[0].derivatives = np.dot(inputs.T,layer1_delta)
        self.layers[0].biases -= learning_rate * np.sum(layer1_delta,axis=0)
        self.layers[0].weights -= learning_rate * self.layers[0].derivatives
        # for layer in reversed(self.layers):
        #     activation = layer.activations
        #     delta = error * layer.activation_function.backward(activation)
        #     layer.derivatives = np.dot(activation.T,delta)
        #     error = np.dot(delta,layer.weights.T)
        # return error

    def gradient_descent(self,learning_rate):
        for layer in self.layers:
            layer.biases -= layer.derivatives
            layer.weights -= layer.derivatives*learning_rate

# test1 = np.random.rand(3000,2)
# test1 /= 2
# targets = np.array([[i[0]+i[1]] for i in test1])
np.random.seed(1337)

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
plt.show()


nn = NeuralNetwork(2,[4],3,"relu")
nn.train(feature_set,one_hot_labels,100,10,0.001)
print("RESULTS: ")
print(nn.predict([0,-3]))
print(nn.predict([2,2]))
print(nn.predict([-2,3]))



# nn.train(test1,targets,100,0.1)
# print(nn.predict([[0.3,0.1],[0.3,0.4],[0.3,0.1],[0.2,0.3]]))

# for layer in nn.layers:
#     print(layer.weights)

#https://www.kdnuggets.com/2019/08/numpy-neural-networks-computational-graphs.html