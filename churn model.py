#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import neccesary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# # Importing the data

# In[2]:


df=pd.read_csv("churn.csv") #load csv file


# # Understanding the data

# In[3]:


df #checkout our data set


# In[4]:


#df.isnull().sum() #check if there is any null values in our dataset


# In[5]:


# col_list=list(df.columns.values) #to understand our dataset more precisily (unique fuction to see how many type of values are)
# for i in range(len(col_list)):
#     print(i,col_list[i],":",df[col_list[i]].unique())


# # Data Manipulation

# In[6]:


#columns like 'PaymentMethod', 'InternetService' needed to get dummies because the is no way to classify them
pay_meth = pd.get_dummies(df['PaymentMethod'])
internet_service = pd.get_dummies(df['InternetService'])


# In[7]:


#concat the columns of dummy variables
df= pd.concat([df,pay_meth], axis=1)


# In[8]:


df= pd.concat([df,internet_service], axis=1)


# In[9]:


#lets drop unwanted columns
df= df.drop(['InternetService','PaymentMethod','customerID' ,'No'], axis=1)


# In[10]:


df


# In[11]:


#classifier or model don't take string values so i will replace Yes by '1' and No by '0'
df = df.replace(to_replace = 'Yes', value=1)

df = df.replace(to_replace = 'No', value=0)

df = df.replace(to_replace = 'Female', value=0)

df = df.replace(to_replace = 'Male', value=1)

# people or customer with 'No internet service' 'No phone service' i will consider there values 'Zero' in there respective columns
df = df.replace(to_replace = 'No internet service', value=0)
df = df.replace(to_replace = 'No phone service', value=0)

# In PaymentMethod i  value them 1 for Month to month, 12 for 1 year, and 24 for 2 years 
df = df.replace(to_replace = 'Month-to-month', value=1)
df = df.replace(to_replace = 'One year', value=12)
df = df.replace(to_replace = 'Two year', value=24)


# In[12]:


df


# In[13]:


col_list_again=list(df.columns.values) #to understand our dataset more precisily (unique fuction to see how many type of values are)
# for i in range(len(col_list_again)):
#     print(i,col_list[i],":",df[col_list[i]].unique())


# # Data Visualization

# In[14]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df.hist(bins=50, figsize= (20, 15)) #take a look at our dataset on histogram


# In[15]:


col_list_again


# In[16]:


corr_matrix = df.corr()
correlation=corr_matrix['Churn'].sort_values(ascending=False)


# In[17]:


correlation


# # Implement Machine Learning Models

# In[30]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

df['TotalCharges'] = df['TotalCharges'].replace(to_replace = " ", value=1394.55)


# In[26]:


X=df.drop(['Churn'],axis=1)
Y=df['Churn']


# In[27]:


X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)


# In[35]:


#scaling
from sklearn.preprocessing import StandardScaler

sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)


# In[40]:


#make Artificial neutral network
from keras.models import Sequential#for model
from keras.layers import Dense#to add preceptrons
from keras.layers import LeakyReLU, PReLU, ELU#for detemining the activation
from keras.layers import Dropout


# In[41]:


model= Sequential()
#adding the imput layer and the first hidden layer
model.add(Dense(units=23, kernel_initializer='he_uniform', activation='relu', input_dim=23))


# In[70]:


#second layer
model.add(Dense(units=15, kernel_initializer= 'he_uniform', activation='relu'))
#third layer
model.add(Dense(units=10, kernel_initializer= 'he_uniform', activation='relu'))
#output layer
model.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))


# In[71]:


model.compile(optimizer= 'Adamax', loss= 'binary_crossentropy', metrics=['accuracy'])


# In[72]:


model_history= model.fit(X_train, Y_train, validation_split=0.33, batch_size=10, epochs=50)


# # Model Evaluation

# In[73]:


print(model_history.history.keys())
#summerize history for accuracy
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[74]:


#summerize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[75]:


y_pred = model.predict(X_test)


# In[76]:


y_pred = (y_pred > 0.5)


# In[77]:


y_pred


# In[78]:


from sklearn.metrics import confusion_matrix


# In[81]:


cm=confusion_matrix(Y_test, y_pred)


# In[82]:


import seaborn as sn
# sn.heatmap(cm, annot= True)
sn.set(font_scale = 2)
sn.heatmap(cm, annot =True, annot_kws={"size":50})

plt.show()


# # Final Conclusions

# In[83]:


from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, Y_test)


# In[84]:


score

 """

