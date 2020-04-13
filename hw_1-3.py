#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from hw_1_3_function import Network
import hw_1_3_function
import matplotlib.pyplot as plt
data = np.load( 'train.npz' )
data1 = np.load('test.npz')

# In[2]:


image_data , label_data = data['image'],data['label']
image_test_data, label_test_data = data1['image'],data1['label']

# In[3]:


x_train = image_data
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]).astype('float32') #28*28 => 1*784
y_train = label_data
y_labels = y_train

x_train = x_train / 255  #normalization

x_test = image_test_data
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]).astype('float32') #28*28 => 1*784
y_test = label_test_data

x_test = x_test / 255  #normalization


# In[4]:


y_train = list(map(int , y_train))


# In[5]:


# y_train => onehot encoded
onehot_encoded = list()
for value in y_train:
    letter = [0 for _ in range(10)]
    letter[value] = 1
    onehot_encoded.append(letter)
y_train_onehot_encoded = onehot_encoded


# In[6]:


net = Network(
                 Neural_network_layer = [784, 20, 2 , 10], 
                 batch_size = 10,
                 num_epochs = 500,
                 learning_rate = 0.001, 
             )

feature_10_epoch,feature_20_epoch , feature_40_epoch , feature_80_epoch , feature_160_epoch , feature_320_epoch , feature_500_epoch = net.train(x_train, y_train_onehot_encoded, y_labels , x_test , y_test)


# In[7]:


def draw_feature(tar, epoch):
    plt.figure(figsize=(5,5))

    for i in range(len(tar)):
        if(tar[i][0] == 0): 
            plt.scatter(tar[i][1][0][0] , tar[i][1][0][1] , s = 10 , c = 'red' )

        elif(tar[i][0] == 1):
            plt.scatter(tar[i][1][0][0] , tar[i][1][0][1] , s = 10 , c = 'blue' )

        elif(tar[i][0] == 2):
            plt.scatter(tar[i][1][0][0] , tar[i][1][0][1] , s = 10 , c = 'green')

        elif(tar[i][0] == 3):
            plt.scatter(tar[i][1][0][0] , tar[i][1][0][1] , s = 10 , c = 'yellow')

        elif(tar[i][0] == 4):
            plt.scatter(tar[i][1][0][0] , tar[i][1][0][1] , s = 10 , c = 'black')

        elif(tar[i][0] == 5):
            plt.scatter(tar[i][1][0][0] , tar[i][1][0][1] , s = 10 , c = 'azure')

        elif(tar[i][0] == 6):
            plt.scatter(tar[i][1][0][0] , tar[i][1][0][1] , s = 10 , c = 'cyan')

        elif(tar[i][0] == 7):
            plt.scatter(tar[i][1][0][0] , tar[i][1][0][1] , s = 10 , c = 'darkgray')

        elif(tar[i][0] == 8):
            plt.scatter(tar[i][1][0][0] , tar[i][1][0][1] , s = 10 , c = 'darkred')

        elif(tar[i][0] == 9):
            plt.scatter(tar[i][1][0][0] , tar[i][1][0][1] , s = 10 , c = 'gold')

    plt.scatter(30 , 0 , c = 'red' ,label = '0')
    plt.scatter(30 , 0 , c = 'blue' ,label = '1')
    plt.scatter(30 , 0 , c = 'green' ,label = '2')
    plt.scatter(30 , 0 , c = 'yellow' ,label = '3')
    plt.scatter(30 , 0 , c = 'black' ,label = '4')
    plt.scatter(30 , 0 , c = 'azure' ,label = '5')
    plt.scatter(30 , 0 , c = 'cyan' ,label = '6')
    plt.scatter(30 , 0 , c = 'darkgray' ,label = '7')
    plt.scatter(30 , 0 , c = 'darkred' ,label = '8')
    plt.scatter(30 , 0 , c = 'gold' ,label = '9')
    
    plt.title('2D feature %s epoch' %epoch)
    plt.legend(loc='upper right')
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.savefig("hw1-3 %s epoch .png" %epoch)
    #plt.show()


# In[8]:


draw_feature(feature_10_epoch , "10")
draw_feature(feature_20_epoch , "20")
draw_feature(feature_40_epoch , "40")
draw_feature(feature_80_epoch , "80")
draw_feature(feature_160_epoch , "160")
draw_feature(feature_320_epoch , "320")
draw_feature(feature_500_epoch , "500")


# In[ ]:




