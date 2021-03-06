#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from hw_1_1_function import Network
import hw_1_1_function
import matplotlib.pyplot as plt
data = np.load( 'train.npz' )
data1 = np.load('test.npz')

# In[2]:


image_data , label_data = data['image'],data['label']
image_test_data, label_test_data = data1['image'],data1['label']

print('image 數量:',image_data.shape)
print('label 數量:',label_data.shape)
print('image_test_data:',image_test_data.shape)
print('label_test_data:',label_test_data.shape)
# In[3]:


#train data:12000 
#test data:5768
all_image_data = image_data 
all_image_data = all_image_data.reshape(all_image_data.shape[0], all_image_data.shape[1]*all_image_data.shape[2]).astype('float32')
all_image_data = all_image_data / 255 #normalization
all_label_data = label_data


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
                 Neural_network_layer = [784, 20, 10], 
                 batch_size = 10,
                 num_epochs = 500,
                 learning_rate = 0.001, 
             )

crosstab_test_data , crosstab_all_data = net.train(x_train, y_train_onehot_encoded, y_labels , x_test , y_test , all_image_data , all_label_data)

print('===== test data crosstab ======')
print(crosstab_test_data)


print('===== train data crosstab ======')
print(crosstab_all_data)

# In[7]:


new_x_axis = np.arange(0,500, 10)




plt.figure(figsize=(20, 5))
plt.subplot(1,3,1)
plt.plot(new_x_axis, np.array(net.training_error_rate))
plt.title('Training error rate')
plt.xlabel('Number of epochs')
plt.ylabel('Error rate')

plt.subplot(1,3,2)
plt.plot(new_x_axis, np.array(net.testing_error_rate))
plt.title('Testing error rate')
plt.xlabel('Number of epochs')
plt.ylabel('Error rate')

plt.subplot(1,3,3)
plt.plot(new_x_axis, np.array(net.training_loss)/100)
plt.title('Training loss')
plt.xlabel('Number of epochs')
plt.ylabel('Average cross entropy')



plt.savefig("hw1-1.png")
plt.show()


# In[ ]:




