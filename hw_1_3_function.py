
import numpy as np

import pandas as pd

# output probability distribution function
def softmax(inputs):
    exp = np.exp(inputs)
    return exp/np.sum(exp, axis = 1, keepdims = True)

# loss
def cross_entropy(inputs, y):
    indices = np.argmax(y, axis = 1).astype(int)
    probability = inputs[np.arange(len(inputs)), indices] #inputs[0, indices]
    log = np.log(probability)
    loss = -1.0 * np.sum(log) / len(log)
    return loss

# L2 regularization
def L2_regularization(la, weight1, weight2, weight3):
    weight1_loss = 0.5 * la * np.sum(weight1 * weight1)
    weight2_loss = 0.5 * la * np.sum(weight2 * weight2)
    weight3_loss = 0.5 * la * np.sum(weight3 * weight3)
    return weight1_loss + weight2_loss + weight3_loss



class Network:
    def __init__(self, 
                 Neural_network_layer, 
                 batch_size,
                 num_epochs,
                 learning_rate, 
                 ):

        self.Neural_network_layer = Neural_network_layer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # build the network
        #         w1/b1    w2/b2   w3/b3 
        #784(inputs) ---> 20(hiden) ---> 2(hiden) ---> 10(output)
        #         x     z1  z2  z3 =y


        #Initial weight & bias
        self.weight1 = np.random.normal(0, 1, [self.Neural_network_layer[0], self.Neural_network_layer[1]])
        self.bias1 = np.zeros((1, self.Neural_network_layer[1]))

        self.weight2 = np.random.normal(0, 1, [self.Neural_network_layer[1], self.Neural_network_layer[2]])
        self.bias2 = np.zeros((1, self.Neural_network_layer[2]))

        self.weight3 = np.random.normal(0, 1, [self.Neural_network_layer[2], self.Neural_network_layer[3]])
        self.bias3 = np.zeros((1, self.Neural_network_layer[3]))

        self.training_loss = []
        self.training_error_rate = []
        self.testing_error_rate = []

        self.feature_10_epoch = []
        self.feature_20_epoch = []
        self.feature_40_epoch = []
        self.feature_80_epoch = []
        self.feature_160_epoch = []
        self.feature_320_epoch = []
        self.feature_500_epoch = []

        
    def train(self, train_data, labels, y1_train , test_data , test_Labels ):

        for epoch in range(self.num_epochs): # training begin
            iteration = 0
            while iteration < len(train_data):

                # batch input
                inputs_batch = train_data[iteration:iteration+self.batch_size]
                labels_batch = labels[iteration:iteration+self.batch_size]
                
                # forward pass
                z1 = np.dot(inputs_batch, self.weight1) + self.bias1
                z2 = np.dot(z1, self.weight2) + self.bias2
                z3 = np.dot(z2, self.weight3) + self.bias3
                y = softmax(z3)

             
                # calculate loss
                loss = cross_entropy(y, labels_batch)
                loss += L2_regularization(0.01, self.weight1, self.weight2, self.weight3)
                
                
                # backward pass
                delta_y = (y - labels_batch) / y.shape[0]
                delta_hidden_layer1 = np.dot(delta_y, self.weight3.T)
                #delta_hidden_layer1[a1 <= 0] = 0 
                delta_hidden_layer = np.dot(delta_hidden_layer1, self.weight2.T)
                #delta_hidden_layer[a2 <= 0] = 0 

                # backpropagation
                weight3_gradient = np.dot(z2.T, delta_y) # forward * backward
                bias3_gradient = np.sum(delta_y, axis = 0, keepdims = True)


                weight2_gradient = np.dot(z1.T, delta_hidden_layer1) 
                bias2_gradient = np.sum(delta_hidden_layer1, axis = 0, keepdims = True)
            
                weight1_gradient = np.dot(inputs_batch.T, delta_hidden_layer)
                bias1_gradient = np.sum(delta_hidden_layer, axis = 0, keepdims = True)

                # L2 regularization
                weight3_gradient += 0.01 * self.weight3
                weight2_gradient += 0.01 * self.weight2
                weight1_gradient += 0.01 * self.weight1


                # stochastic gradient descent
                #update weight and bias
                self.weight1 -= self.learning_rate * weight1_gradient 
                self.bias1 -= self.learning_rate * bias1_gradient
                self.weight2 -= self.learning_rate * weight2_gradient
                self.bias2 -= self.learning_rate * bias2_gradient
                self.weight3 -= self.learning_rate * weight3_gradient
                self.bias3 -= self.learning_rate * bias3_gradient


                iteration += self.batch_size

        
            if(epoch % 10 == 0):

                print("====== Epoch: {:d}/{:d} ======\nLoss: {:.2f}".format(epoch, self.num_epochs, loss))
                acc = self.calculate_acc_rate(train_data,y1_train,"train")
                acc1 = self.calculate_acc_rate(test_data, test_Labels,"test")

                self.training_loss.append(loss)
                self.training_error_rate.append(1-acc)
                self.testing_error_rate.append(1-acc1)
            if(epoch == 10):
                self.feature_10_epoch = self.feature_epoch_array(test_data , test_Labels)

            elif(epoch == 20 ):
                self.feature_20_epoch = self.feature_epoch_array(test_data , test_Labels)

            elif(epoch == 40):
                self.feature_40_epoch = self.feature_epoch_array(test_data , test_Labels)

            elif(epoch == 80):
                self.feature_80_epoch = self.feature_epoch_array(test_data , test_Labels)

            elif(epoch == 160):
                self.feature_160_epoch = self.feature_epoch_array(test_data , test_Labels)

            elif(epoch == 320):
                self.feature_320_epoch = self.feature_epoch_array(test_data , test_Labels)

        self.feature_500_epoch = self.feature_epoch_array(test_data , test_Labels)

        return self.feature_10_epoch , self.feature_20_epoch , self.feature_40_epoch , self.feature_80_epoch , self.feature_160_epoch , self.feature_320_epoch , self.feature_500_epoch



    def feature_epoch_array(self, test_data , test_Labels):
        g = []
        for i in range(len(test_data)):
            l = []
            x = self.latent_features(test_data[i] , test_Labels[i])
            l.append(test_Labels[i])
            l.append(x)
            g.append(l)
        return g

    def latent_features(self, test_data , test_Labels):
        input_layer = np.dot(test_data, self.weight1)
        hidden_layer = input_layer + self.bias1

        hidden_layer1 = np.dot(hidden_layer, self.weight2) 
        output_layer1 = hidden_layer1 + self.bias2

        return output_layer1

    def calculate_acc_rate(self, test_data, labels, train_or_test):
        input_layer = np.dot(test_data, self.weight1)
        hidden_layer = input_layer + self.bias1
        # hidden_layer = relu(input_layer + self.bias1)

        hidden_layer1 = np.dot(hidden_layer, self.weight2)
        hidden_layer1_plus = hidden_layer1 + self.bias2
        # hidden_layer1_relu = relu(hidden_layer1 + self.bias2)

        scores = np.dot(hidden_layer1_plus, self.weight3) + self.bias3

        probs = softmax(scores)
        acc = float(np.sum(np.argmax(probs, 1) == labels)) / float(len(labels))
        print("{:s} error_rate: {:.4f}".format(train_or_test , 1 - acc))
        return acc


