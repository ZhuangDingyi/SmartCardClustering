
# coding: utf-8

# 
# 	1. 经纬度（2）
# 	2. 时间（1）
# 	3. 总出行量（出现在boarding+alighting中的总和）（1）
# 	4. 乘客总分布（3）
# 	5. 从其他车站流入（157）
# 	6. 流入其他车站（157）
# 	7. 中转量（1）
# 
# 		1. 中转率（1）
# 		2. 总转入率（1）
# 		3. 总转出率（1）
# 		4. 总中转乘客类型（3）
# 		5. 地铁-地铁中转量（1）
# 
# 			1. 转入率（1）
# 
# 				1. 转入乘客分布（3）
# 			2. 转出率（1）
# 
# 				1. 转出乘客分布（3）
# 		6. 地铁-公交中转量（1）
# 
# 			1. 转入率（1）
# 
# 				1. 转入乘客分布（3）
# 			2. 转出率（1）
# 
# 				1. 转出乘客分布（3）
# 	8. 地铁站周围公交站的数量（最小travel distance来定）（1）
# 	1. 地铁站周围不同POI的数量？
# 
# 		1. Google Map一次只能返回周围最多200个POI，新加坡这么密集的地方最好半径还是去500比较好，不然差异不是很明显
# 		2. POI种类划分
# 
# 			1. establishment
# 			2. finance
# 			3. food
# 			4. bus_station
# 			5. transit_station
# 			6. place_of_worship
# 			7. health
# 			8. supermarket 2 7 7 8
# 			9. shopping_mall 0 4 13 15
# 			10. store
# 
# 				1. convenience_store （住宅区更高一些？）
# 				2. electronics_store
# 			11. storage
# 			12. school
# 			13. parking
# 			14. park
# 			15. political
# 	2. 线路汇聚性：地铁站是否换乘其他站（1）
# 	3. 作为始发站的运行总距离（1）
# 	4. 作为终点站的运行总距离（1）
# 	5. one-hot表示天数（1 234 5 6 7）（5）
# 	6. one-hot表示时间段（7）

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
import tensorflow as tf
get_ipython().magic('matplotlib inline')

# Timestamps range
# 5-7 pre-morning peak,7-10 morning peak, 10-16 morning off-peak, 
# 16-17 pre-evening peak, 17-19 evening peak,19-22 late evening peak, 22 evening off-peak,
# The above 7 time intervals will be transformed in to a 0-6 catagorical number, then turned into one-hots

ts_0319_s=time.mktime(time.strptime('2012-03-19 00:00:00','%Y-%m-%d %H:%M:%S'))
ts_0319_0=time.mktime(time.strptime('2012-03-19 06:59:59','%Y-%m-%d %H:%M:%S'))
ts_0319_1=time.mktime(time.strptime('2012-03-19 09:59:59','%Y-%m-%d %H:%M:%S'))
ts_0319_2=time.mktime(time.strptime('2012-03-19 15:59:59','%Y-%m-%d %H:%M:%S'))
ts_0319_3=time.mktime(time.strptime('2012-03-19 16:59:59','%Y-%m-%d %H:%M:%S'))
ts_0319_4=time.mktime(time.strptime('2012-03-19 18:59:59','%Y-%m-%d %H:%M:%S'))
ts_0319_5=time.mktime(time.strptime('2012-03-19 21:59:59','%Y-%m-%d %H:%M:%S'))
ts_0319_e=time.mktime(time.strptime('2012-03-19 23:59:59','%Y-%m-%d %H:%M:%S'))

ts_0320_s=time.mktime(time.strptime('2012-03-20 00:00:00','%Y-%m-%d %H:%M:%S'))
ts_0320_0=time.mktime(time.strptime('2012-03-20 06:59:59','%Y-%m-%d %H:%M:%S'))
ts_0320_1=time.mktime(time.strptime('2012-03-20 09:59:59','%Y-%m-%d %H:%M:%S'))
ts_0320_2=time.mktime(time.strptime('2012-03-20 15:59:59','%Y-%m-%d %H:%M:%S'))
ts_0320_3=time.mktime(time.strptime('2012-03-20 16:59:59','%Y-%m-%d %H:%M:%S'))
ts_0320_4=time.mktime(time.strptime('2012-03-20 18:59:59','%Y-%m-%d %H:%M:%S'))
ts_0320_5=time.mktime(time.strptime('2012-03-20 21:59:59','%Y-%m-%d %H:%M:%S'))
ts_0320_e=time.mktime(time.strptime('2012-03-20 23:59:59','%Y-%m-%d %H:%M:%S'))

ts_0321_s=time.mktime(time.strptime('2012-03-21 00:00:00','%Y-%m-%d %H:%M:%S'))
ts_0321_0=time.mktime(time.strptime('2012-03-21 06:59:59','%Y-%m-%d %H:%M:%S'))
ts_0321_1=time.mktime(time.strptime('2012-03-21 09:59:59','%Y-%m-%d %H:%M:%S'))
ts_0321_2=time.mktime(time.strptime('2012-03-21 15:59:59','%Y-%m-%d %H:%M:%S'))
ts_0321_3=time.mktime(time.strptime('2012-03-21 16:59:59','%Y-%m-%d %H:%M:%S'))
ts_0321_4=time.mktime(time.strptime('2012-03-21 18:59:59','%Y-%m-%d %H:%M:%S'))
ts_0321_5=time.mktime(time.strptime('2012-03-21 21:59:59','%Y-%m-%d %H:%M:%S'))
ts_0321_e=time.mktime(time.strptime('2012-03-21 23:59:59','%Y-%m-%d %H:%M:%S'))

ts_0322_s=time.mktime(time.strptime('2012-03-22 00:00:00','%Y-%m-%d %H:%M:%S'))
ts_0322_0=time.mktime(time.strptime('2012-03-22 06:59:59','%Y-%m-%d %H:%M:%S'))
ts_0322_1=time.mktime(time.strptime('2012-03-22 09:59:59','%Y-%m-%d %H:%M:%S'))
ts_0322_2=time.mktime(time.strptime('2012-03-22 15:59:59','%Y-%m-%d %H:%M:%S'))
ts_0322_3=time.mktime(time.strptime('2012-03-22 16:59:59','%Y-%m-%d %H:%M:%S'))
ts_0322_4=time.mktime(time.strptime('2012-03-22 18:59:59','%Y-%m-%d %H:%M:%S'))
ts_0322_5=time.mktime(time.strptime('2012-03-22 21:59:59','%Y-%m-%d %H:%M:%S'))
ts_0322_e=time.mktime(time.strptime('2012-03-22 23:59:59','%Y-%m-%d %H:%M:%S'))

ts_0323_s=time.mktime(time.strptime('2012-03-23 00:00:00','%Y-%m-%d %H:%M:%S'))
ts_0323_0=time.mktime(time.strptime('2012-03-23 06:59:59','%Y-%m-%d %H:%M:%S'))
ts_0323_1=time.mktime(time.strptime('2012-03-23 09:59:59','%Y-%m-%d %H:%M:%S'))
ts_0323_2=time.mktime(time.strptime('2012-03-23 15:59:59','%Y-%m-%d %H:%M:%S'))
ts_0323_3=time.mktime(time.strptime('2012-03-23 16:59:59','%Y-%m-%d %H:%M:%S'))
ts_0323_4=time.mktime(time.strptime('2012-03-23 18:59:59','%Y-%m-%d %H:%M:%S'))
ts_0323_5=time.mktime(time.strptime('2012-03-23 21:59:59','%Y-%m-%d %H:%M:%S'))
ts_0323_e=time.mktime(time.strptime('2012-03-23 23:59:59','%Y-%m-%d %H:%M:%S'))

ts_0324_s=time.mktime(time.strptime('2012-03-24 00:00:00','%Y-%m-%d %H:%M:%S'))
ts_0324_0=time.mktime(time.strptime('2012-03-24 06:59:59','%Y-%m-%d %H:%M:%S'))
ts_0324_1=time.mktime(time.strptime('2012-03-24 09:59:59','%Y-%m-%d %H:%M:%S'))
ts_0324_2=time.mktime(time.strptime('2012-03-24 15:59:59','%Y-%m-%d %H:%M:%S'))
ts_0324_3=time.mktime(time.strptime('2012-03-24 16:59:59','%Y-%m-%d %H:%M:%S'))
ts_0324_4=time.mktime(time.strptime('2012-03-24 18:59:59','%Y-%m-%d %H:%M:%S'))
ts_0324_5=time.mktime(time.strptime('2012-03-24 21:59:59','%Y-%m-%d %H:%M:%S'))
ts_0324_e=time.mktime(time.strptime('2012-03-24 23:59:59','%Y-%m-%d %H:%M:%S'))

ts_0325_s=time.mktime(time.strptime('2012-03-25 00:00:00','%Y-%m-%d %H:%M:%S'))
ts_0325_0=time.mktime(time.strptime('2012-03-25 06:59:59','%Y-%m-%d %H:%M:%S'))
ts_0325_1=time.mktime(time.strptime('2012-03-25 09:59:59','%Y-%m-%d %H:%M:%S'))
ts_0325_2=time.mktime(time.strptime('2012-03-25 15:59:59','%Y-%m-%d %H:%M:%S'))
ts_0325_3=time.mktime(time.strptime('2012-03-25 16:59:59','%Y-%m-%d %H:%M:%S'))
ts_0325_4=time.mktime(time.strptime('2012-03-25 18:59:59','%Y-%m-%d %H:%M:%S'))
ts_0325_5=time.mktime(time.strptime('2012-03-25 21:59:59','%Y-%m-%d %H:%M:%S'))
ts_0325_e=time.mktime(time.strptime('2012-03-25 23:59:59','%Y-%m-%d %H:%M:%S'))

tint_0=ts_0319_0-ts_0319_s
tint_1=ts_0319_1-ts_0319_s
tint_2=ts_0319_2-ts_0319_s
tint_3=ts_0319_3-ts_0319_s
tint_4=ts_0319_4-ts_0319_s
tint_5=ts_0319_5-ts_0319_s
tint_6=ts_0319_e-ts_0319_s
tint=[0,tint_0,tint_1,tint_2,tint_3,tint_4,tint_5,tint_6]


# In[2]:


dayDict={'Monday':ts_0319_s,'Tuesday':ts_0320_s,'Wednesday':ts_0321_s,'Thursday':ts_0322_s,'Friday':ts_0323_s,
         'Saturday':ts_0324_s,'Sunday':ts_0325_s}
passengerDict={'Adult':'adu_dis','Child/Student':'chd_dis','Senior Citizen':'sen_dis'}
def getDayTimeInterval(day,time):
    ts=dayDict[day]
    interval=time-ts
    for i in range(len(tint)-1):
        if tint[i]<interval<=tint[i+1]:
            return i
    #return -1 # not in range
    if interval<=tint[0]:
        return 0
    if interval>tint[6]:
        return 6


# In[11]:


MRTStops=pd.read_csv('MRTStops_Geocoded.csv')
print(MRTStops.head())
stationDict=MRTStops['MRTStopsName']
stationDict=stationDict.to_dict()
print(stationDict)
stationDict_re=dict(zip(stationDict.values(),stationDict.keys()))

def getFeatMatRowIdx(MRTStopsName,interval):
    staNum=stationDict_re[MRTStopsName]
    return staNum+interval*len(stationDict_re)









# # Begin to train the reduced representation of flow information

# In[3]:


flowMat_dict=joblib.load('flowMat_dict_onehot.asv')
#train_data_test=tf.data.Dataset.from_tensor_slices(flowMat_dict[0])
train_data_test=flowMat_dict[0]
learning_rate = 1
training_epochs = 10**2
batch_size = 10
display_step = 1
op_step = training_epochs/100
n_input=len(flowMat_dict[0])
n_dim=np.shape(flowMat_dict[0])[1]
print(n_input,n_dim)


# In[15]:


# Test the result if the input is the flow matrix
latlng2stack=np.column_stack((MRTStops['MRTLat'],MRTStops['MRTLng']))
station_latlng_854=np.zeros((len(dayDict)*len(MRTStops),2))
station_latlng_854[0:122,:]=latlng2stack
station_latlng_854[122:(122*2),:]=latlng2stack
station_latlng_854[(122*2):(122*3),:]=latlng2stack
station_latlng_854[(122*3):(122*4),:]=latlng2stack
station_latlng_854[(122*4):(122*5),:]=latlng2stack
station_latlng_854[(122*5):(122*6),:]=latlng2stack
station_latlng_854[(122*6):(122*7),:]=latlng2stack
print(station_latlng_854[0],station_latlng_854[122])

flowMat_dict=joblib.load('flowMat_dict_onehot.asv')
#train_data_test=tf.data.Dataset.from_tensor_slices(flowMat_dict[0])
train_data_test=np.column_stack((station_latlng_854,flowMat_dict[0]))
norm_test=np.linalg.norm(train_data_test,ord=1,axis=0,keepdims=True)
train_data_test=train_data_test/norm_test
train_data_test[np.isnan(train_data_test)]=0
learning_rate = 1e-6
training_epochs = 5*10**5
batch_size = 10
display_step = 1
op_step = training_epochs/100
n_input=len(train_data_test)
n_dim=np.shape(train_data_test)[1]
print(n_input,n_dim)

 # tf Graph input (only pictures)  
X = tf.placeholder("float", [None, n_dim]) 
 # 用字典的方式存储各隐藏层的参数  
n_hidden_1 = 200 # 第一编码层神经元个数  
n_hidden_2 = 100 # 第二编码层神经元个数
n_hidden_3 = 50
# 权重和偏置的变化在编码层和解码层顺序是相逆的  
# 权重参数矩阵维度是每层的 输入*输出，偏置参数维度取决于输出层的单元数  
weights = {  
    'encoder_h1': tf.Variable(tf.random_normal([n_dim, n_hidden_1])),  
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),  
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_dim])),
}  
biases = {  
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),  
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b3': tf.Variable(tf.random_normal([n_dim])),  
} 

 # 每一层结构都是 xW + b  
# 构建编码器  
def encoder(x):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),
                                 biases['encoder_b1']))
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_h2']),
                                 biases['encoder_b2']))
    layer_3=tf.nn.sigmoid(tf.add(tf.matmul(layer_2,weights['encoder_h3']),
                                 biases['encoder_b3']))
    return layer_3

def decoder(x):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h1']),
                                 biases['decoder_b1']))
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['decoder_h2']),
                                 biases['decoder_b2']))
    layer_3=tf.nn.sigmoid(tf.add(tf.matmul(layer_2,weights['decoder_h3']),
                                 biases['decoder_b3']))
    return layer_3

encoder_op=encoder(X)
decoder_op=decoder(encoder_op)

# 预测  
y_pred = decoder_op  
y_true = X 

#cost and optimizer
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) #最小二乘法  
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

loss=[]
saver=tf.train.Saver()
with tf.Session() as sess:
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  
        init = tf.initialize_all_variables()  
    else:  
        init = tf.global_variables_initializer()  
    sess.run(init)
    # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练
    #total_batch=int(n_input/batch_size)
    #batches=tf.train.batch(train_data_test,batch_size=batch_size)
    #coord=tf.train.Coordinator()
    #threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    for epoch in tqdm_notebook(range(training_epochs),'epoches'):
        #for i in tqdm_notebook(range(total_batch),'iteral total_batch'):
            #batch_xs=tf.data.Iterator(train_data_test.shuffle(1000).batch(batch_size))
            #iterator=train_data_test.make_one_shot_iterator()
            
            #batch_xs=batch_xs.reshape(1,-1)
            #d,c=sess.run([optimizer,cost],feed_dict={X:batch_xs})
        #for i in range(len(train_data_test)):
            #batch_xs=train_data_test[i]
            #batch_xs=batch_xs.reshape(1,-1)
        batch_xs=train_data_test
        d,c=sess.run([optimizer,cost],feed_dict={X:batch_xs})
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "{:.9f}".format(c))
        if epoch % op_step ==0:
            loss.append(c)
            saver.save(sess,'model_0730/AETest.ckpt',global_step=epoch+1)
    print("Optimization finished")
    plt.figure(1)
    plt.plot(loss)
    encoded_res=sess.run(encoder_op,feed_dict={X:train_data_test})
    x_tsne=TSNE(n_components=2,learning_rate=100).fit_transform(encoded_res)
    plt.figure(2)
    plt.scatter(x_tsne[:,0],x_tsne[:,1])
    #plt.colorbar()
    plt.show()


# In[8]:


sess1=tf.Session()
model_file=tf.train.latest_checkpoint('model_0728/')
tf.train.Saver.restore(sess1,model_file)
encoded_res=sess1.run(encoder_op,feed_dict={X:train_data_test})
x_tsne=TSNE(n_components=2,learning_rate=100).fit_transform(encoded_res)
plt.figure(2)
plt.scatter(x_tsne[:,0],x_tsne[:,1])
#plt.colorbar()
plt.show()



# In[113]:


encoded_res=sess1.run(encoder_op,feed_dict={X:train_data_test})
#print(encoded_res[0])
decoded_res=sess1.run(decoder_op,feed_dict={X:train_data_test})
#print(decoded_res[0])
print(decoded_res[0]*norm_test)
dev_0=decoded_res*norm_test-train_data_test*norm_test
plt.plot(dev_0)


# In[116]:


print(np.shape(dev_0),np.shape(decoded_res))
print(np.mean(norm_test))
print(np.max(norm_test))
print(np.mean(dev_0))
print(np.max(dev_0))


# In[117]:


# Plot the deviation
# Only several dimensions have inflence the optimization
for j in range(len(dev_0)):
    plt.plot(dev_0[j])
plt.savefig('fig/deviation-after-reduce-flowmat.png')
plt.show()



# <font color=red size=42> Maybe we could further analyze why these stations are outliers </font>
# 

# # Considering use denoised auto-encoder

# In[109]:




