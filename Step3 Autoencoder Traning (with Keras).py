
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import csv
import time,datetime
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook
import pickle
from sklearn.externals import joblib
from scipy import spatial
from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
import sklearn
import pickle

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
import keras
import keras.backend as K
from keras_tqdm import TQDMNotebookCallback

import tensorflow as tf
#get_ipython().magic('matplotlib inline')

#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
        #self.dev_metric= {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        #self.dev_metric['batch'].append(logs.get('dev_metric'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        #self.dev_metric['epoch'].append(logs.get('dev_metric'))
        
    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
            # deviation
            #plt.plot(iters, self.dev_metric[loss_type], 'y', label='deviation')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

# # Construct auto-encoder for dimension reduction among 7 time intervals(7 models)

# In[2]:

flowDict=joblib.load('flowMat_dict_onehot.asv')
np.shape(flowDict[0])


# In[3]:

# Construct dictionary for differnt time of the day
mat_sta_id=np.eye(122)
for i in range(6):
    tmp=np.eye(122)
    mat_sta_id=np.row_stack((mat_sta_id,tmp))
del tmp


for tint in range(7):
    eq0='flowMat_tint_'+str(tint)+'=np.empty(shape=[0,373])' # The feature is 244+7 adding 122 dimension one-hot for station identification
    exec(eq0)

del tint,eq0

cnt=0
for day in ['0319','0320','0321','0322','0323','0324','0325']:
    eq1='flowMat_'+day+'=flowDict['+str(cnt)+']'
    eq2='tmp=np.column_stack((mat_sta_id,flowMat_'+day+ '[:,:-7]))'
    exec(eq1)
    exec(eq2)
    
    # flowMat has 122*7 rows, from Monday to Sunday
    flowMat_tint_0=np.append(flowMat_tint_0,tmp[0:122,:],axis=0)
    flowMat_tint_1=np.append(flowMat_tint_1,tmp[122:(122*2),:],axis=0)
    flowMat_tint_2=np.append(flowMat_tint_2,tmp[(122*2):(122*3),:],axis=0)
    flowMat_tint_3=np.append(flowMat_tint_3,tmp[(122*3):(122*4),:],axis=0)
    flowMat_tint_4=np.append(flowMat_tint_4,tmp[(122*4):(122*5),:],axis=0)
    flowMat_tint_5=np.append(flowMat_tint_5,tmp[(122*5):(122*6),:],axis=0)
    flowMat_tint_6=np.append(flowMat_tint_6,tmp[(122*6):(122*7),:],axis=0)
    
    cnt+=1
del cnt
flowDict_int={}
for i in range(7):
    eq3='flowDict_int['+str(i)+']=flowMat_tint_'+str(i)
    exec(eq3)
joblib.dump(flowDict_int,'flowMat_dict_tint_stn_onehot.asv')


# In[4]:

# Begin to train 7 separate models 
# Take the first as example
flowDict_int=joblib.load('flowMat_dict_tint_stn_onehot.asv')
train_data_tint_0=flowDict_int[0]

n_dim=np.shape(train_data_tint_0)[1]
n_samples=np.shape(train_data_tint_0)[0]
# Normalization

# Mean-std normalization ( with poor performance)
#norm_mean_0=np.mean(train_data_tint_0,axis=0)
#norm_std_0=np.std(train_data_tint_0,axis=0)
#train_data_tint_0=(train_data_tint_0-norm_mean_0)/norm_std_0
#train_data_tint_0=np.nan_to_num(train_data_tint_0)

# Max normalization
'''
train_data_tint_0=flowDict_int[0]
train_data_tint_0,norm_tint_0=sklearn.preprocessing.normalize(train_data_tint_0,norm='max',return_norm=True,axis=0)
norm_tint_0=np.linalg.norm(train_data_tint_0,ord=1,axis=0,keepdims=True)
train_data_tint_0=train_data_tint_0/norm_tint_0
train_data_tint_0[np.isnan(train_data_tint_0)]=0
                  
train_data_tint_1=flowDict_int[1]
train_data_tint_1,norm_tint_1=sklearn.preprocessing.normalize(train_data_tint_1,norm='max',return_norm=True,axis=0)
norm_tint_1=np.linalg.norm(train_data_tint_1,ord=1,axis=0,keepdims=True)
train_data_tint_1=train_data_tint_1/norm_tint_1
train_data_tint_1[np.isnan(train_data_tint_1)]=0
                  
train_data_tint_2=flowDict_int[2]
train_data_tint_2,norm_tint_2=sklearn.preprocessing.normalize(train_data_tint_2,norm='max',return_norm=True,axis=0)
norm_tint_2=np.linalg.norm(train_data_tint_2,ord=1,axis=0,keepdims=True)
train_data_tint_2=train_data_tint_2/norm_tint_2
train_data_tint_2[np.isnan(train_data_tint_2)]=0            

train_data_tint_3=flowDict_int[3]
train_data_tint_3,norm_tint_3=sklearn.preprocessing.normalize(train_data_tint_3,norm='max',return_norm=True,axis=0)
norm_tint_3=np.linalg.norm(train_data_tint_3,ord=1,axis=0,keepdims=True)
train_data_tint_3=train_data_tint_3/norm_tint_3
train_data_tint_3[np.isnan(train_data_tint_3)]=0  

train_data_tint_4=flowDict_int[4]
train_data_tint_4,norm_tint_4=sklearn.preprocessing.normalize(train_data_tint_4,norm='max',return_norm=True,axis=0)
#norm_tint_4=np.linalg.norm(train_data_tint_4,ord=1,axis=0,keepdims=True)
#train_data_tint_4=train_data_tint_4/norm_tint_4
#train_data_tint_4[np.isnan(train_data_tint_4)]=0  
'''
train_data_tint_5=flowDict_int[5]
train_data_tint_5,norm_tint_5=sklearn.preprocessing.normalize(train_data_tint_5,norm='max',return_norm=True,axis=0)                   

flowMat_dict=joblib.load('flowMat_dict_onehot.asv')
#train_data_test=tf.data.Dataset.from_tensor_slices(flowMat_dict[0])
#train_data_test=np.column_stack((mat_sta_id,flowMat_dict[0][:,:-7]))
train_data_test=(flowMat_dict[0])
train_data_test=np.column_stack((mat_sta_id,train_data_test))
norm_test=np.linalg.norm(train_data_test,ord=1,axis=0,keepdims=True)
train_data_test=train_data_test/norm_test
train_data_test[np.isnan(train_data_test)]=0
n_dim_test=np.shape(train_data_test)[1]
n_samples_test=np.shape(train_data_test)[0]


# In[9]:

#print(train_data_tint_0[0])


# In[63]:

#np.shape(train_data_tint_0.reshape([-1,1]))
#np.shape(norm_mean.reshape([-1,1]))
#np.shape(train_data_tint_0-norm_mean)
#norm_std_0
#(train_data_tint_0-norm_mean_0)[0]
#print(train_data_tint_0)
#print(train_data_tint_0[0])
#print(train_data_tint_0[1])
#print(train_data_tint_0[2])



# In[10]:

# Build the autoencoder model
#---- Hyper parameters


# Seems like the loss have not fully converged


# In[117]:

# construct the autoencoder model
encoding_dim = 10
encode_layer1=256
encode_layer2=128
encode_layer3=64
encode_layer4=30

decode_layer1=encode_layer4
decode_layer2=encode_layer3
decode_layer3=encode_layer2
decode_layer4=encode_layer1

# this is our input placeholder
input_data = Input(shape=(n_dim,))


# encoder layers
encoded = Dense(encode_layer1, activation='relu')(input_data)
encoded = Dense(encode_layer2, activation='relu')(encoded)
encoded = Dense(encode_layer3, activation='relu')(encoded)
encoded = Dense(encode_layer4, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

# decoder layers
decoded = Dense(decode_layer1, activation='relu')(encoder_output)
decoded = Dense(decode_layer2, activation='relu')(decoded)
decoded = Dense(decode_layer3, activation='relu')(decoded)
decoded = Dense(decode_layer4, activation='relu')(decoded)
decoded = Dense(n_dim, activation='tanh')(decoded)

autoencoder = Model(input=input_data, output=decoded)
# construct the encoder model for clustering and plotting
encoder = Model(input=input_data, output=encoder_output)

ada=keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)
history_tint_5 = LossHistory()
checkpoint_tint_5=keras.callbacks.ModelCheckpoint(filepath='model_0801/checkpoint_tint_5/checkpoint-{epoch:05d}-tint-5.hdf5',
                                                  monitor='val_loss',save_best_only=False,mode='auto',verbose=1,period=10**3)

def dev_metric(y_true, y_pred):
    return K.mean(K.abs(y_pred-y_true))

# compile autoencoder
#autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(loss='mse',optimizer=ada)

#autoencoder=keras.models.load_model('model_0801/checkpoint_tint_5/checkpoint-138000-tint-5.hdf5')
# training
autoencoder.fit(train_data_tint_5, train_data_tint_5,
                nb_epoch=10**5,
                shuffle=True,
                batch_size=256,
                verbose=1,
                callbacks=[history_tint_5,checkpoint_tint_5])

history_tint_5.loss_plot('epoch')
history_tint_5.loss_plot('batch')
autoencoder.save('model_0801/autoencoder_tint_5_4layers_100000.h5')
encoder.save('model_0801/encoder_tint_5_4layers_100000.h5')
#joblib.dump(history_tint_0,'model_0801/history.h5')
#out_his=open('model_0801/history_tint_0.pkl','wb')
joblib.dump(history_tint_5.losses,'model_0801/history_tint_5_loss.asv')
# In[110]:

#autoencoder_test.save('model_0801/autoencoder_previous_train_data_100000.h5')
#history.loss_plot('batch'
print(np.shape(train_data_test))

dev_0=(autoencoder.predict(train_data_tint_5)-train_data_tint_5)
for i in range(len(dev_0)):
    plt.plot(dev_0[i])
plt.show()
                         
# In[108]:
# In[99]:

#sgd=keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
#adm=keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)





# In[71]:



