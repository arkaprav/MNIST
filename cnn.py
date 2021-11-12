#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
(X_train, y_train), (X_test,y_test) = datasets.mnist.load_data()
X_train = tf.expand_dims(X_train, axis=-1)
X_test = tf.expand_dims(X_test, axis=-1)
X_train.shape


# In[2]:


X_test.shape


# In[3]:


y_train.shape


# In[4]:


y_train[:5]


# In[5]:


y_train = y_train.reshape(-1,)
y_train[:5]


# In[6]:


y_test = y_test.reshape(-1,)
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[7]:


def plot_sample(X, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(y[index])


# In[8]:


plot_sample(X_train, y_train, 0)


# In[9]:


plot_sample(X_train, y_train, 1)


# In[11]:


X_train = X_train / 255
X_test = X_test / 255


# In[12]:


ann = models.Sequential([
        layers.Flatten(input_shape=[28,28,1]),
        layers.Dense(4000, activation='relu'),
        layers.Dense(3000, activation='relu'),
        layers.Dense(10, activation='softmax')    
    ])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=10)


# In[13]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))


# In[14]:


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[15]:


cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[16]:


cnn.fit(X_train, y_train, epochs=10)


# In[17]:


cnn.evaluate(X_test,y_test)


# In[18]:


y_pred = cnn.predict(X_test)
y_pred[:5]


# In[19]:


y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


# In[20]:


y_test[:5]


# In[21]:


plot_sample(X_test, y_test,3)


# In[24]:


y_classes[3]

